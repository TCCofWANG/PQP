# import torch
# from jsq.smooth import smooth_layer
# from jsq.quantize import quantize_layer
# from jsq.data import get_loaders
from jsq.layerwrapper import WrappedGPT
from jsq.utils import find_layers, prepare_calibration_input, clip_matrix, generate_ss
import torch

import time
from utils.gptq import GPTQ
from utils.quant import Quant, quant

from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
    pseudo_quantize_tensor,
)
from utils.tools import *

def joint_pq(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # prune_start = time.time()
    CHATGLM = False
    Falcon = False
    # if hasattr(model.model, "transformer"):
    #     if hasattr(model.model.transformer, "embedding"):
    #         CHATGLM = True
    #     elif hasattr(model.model.transformer, "word_embeddings"):
    #         Falcon = True
    #
    # # print(model)
    # # print("loading calibdation data")
    # # dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    # # print("dataset loading complete")
    # with torch.no_grad():
    #     inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    #
    # if CHATGLM:
    #     layers = model.model.transformer.encoder.layers
    # elif Falcon:
    #     layers = model.model.transformer.h
    # else:
    #     layers = model.model.model.layers

    prune_start = time.time()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if 'llama' in args.model.lower():
        layers = model.model.model.layers
        model.model.model.embed_tokens = model.model.model.embed_tokens.to(device)
    elif "opt" in args.model.lower():
        layers = model.model.model.decoder.layers
        model.model.model.decoder.embed_tokens = model.model.model.decoder.embed_tokens.to(device)
        model.model.model.decoder.embed_positions = model.model.model.decoder.embed_positions.to(device)

    dtype = next(iter(model.model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)  # ori: 128
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model.lower():
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = layers[0].to(device)
    layers[0] = Catcher(layers[0])
    for i in range(args.nsamples):
        try:
            batch = dataloader[i]
            model.model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers = layers.cpu()
    if 'llama' in args.model.lower():
        model.model.model.embed_tokens = model.model.model.embed_tokens.cpu()
        position_ids = cache['position_ids']
    elif 'opt' in args.model.lower():
        model.model.model.decoder.embed_tokens = model.model.model.decoder.embed_tokens.cpu()
        model.model.model.decoder.embed_positions = model.model.model.decoder.embed_positions.cpu()
        position_ids = None

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    torch.cuda.empty_cache()

    # if args.quant_type == 'awq':
    #     q_config = {
    #         "zero_point": not args.no_zero_point,  # by default True
    #         "q_group_size": args.q_group_size,  # whether to use group quantization
    #     }
    #     print("Quantization config:", q_config)
    #
    #     # 如果指定了加载预先计算好的 AWQ 结果
    #     if args.load_awq:
    #         print("Loading pre-computed AWQ results from", args.load_awq)
    #         # 加载 scale 和 clip 参数
    #         awq_results = torch.load(args.load_awq, map_location="cpu")
    #         apply_awq(model.model, awq_results)


    for i in range(len(layers)):
        layer = layers[i].to(device)
        use_old_forward(layer, recurse=True)
        layer_name = f'model.layers.{i}'

        subset = find_layers(layer)

        if f"model.model.layers.{i}" in model.model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.model.hf_device_map[f"model.model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        print(f"get scales of layer {i} and pruning")
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()

            if name in act_scales:
                act_scales[layer_name + '.' + name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[layer_name + '.' + name] = comming_max

        def add_batch(name):
            def tmp(_, inp, out):
                inp = clip_matrix(inp[0].data, args.abs, 0, args.clip_h)
                stat_tensor(name, inp)
                wrapped_layers[name].add_batch(inp, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                elif Falcon:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=None)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            weight = torch.abs(subset[name].weight.data)
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            ss = generate_ss(wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num, subset[name].weight.data)
            W_metric = weight * activation
            W_metric = W_metric + args.rho * ss

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                elif Falcon:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=None)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        # print(f"smoothing layer {i}")
        # smooth_layer(layer_name, layer, act_scales, 0.5)
        #
        # print(f"quantizing layer {i}")
        # quantize_layer(layer, nbits=args.nbits)

        if args.quant_type == 'rtn':
            for name in subset:
                print('RTN quantization.....')
                quantizer = Quant()
                quantizer.configure(args.wbits, perchannel=True, sym=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quant(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                    next(iter(layer.parameters())).dtype)
        elif args.quant_type == 'gptq':
            print(f"GPTQ quantizing layer {i}")

            full = subset
            print(f"---------------- GPTQ Layer {i} of {len(layers)} ----------------")
            if args.true_sequential:
                sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'],
                              ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
            else:
                sequential = [list(full.keys())]

            for names in sequential:
                gptq_subset = {n: full[n] for n in names}

                gptq = {}
                for name in gptq_subset:
                    gptq[name] = GPTQ(gptq_subset[name])
                    gptq[name].quantizer = Quant()
                    gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

                def gptq_add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data)

                    return tmp

                handles = []
                for name in gptq_subset:
                    handles.append(gptq_subset[name].register_forward_hook(gptq_add_batch(name)))
                for j in range(args.nsamples):
                    with torch.no_grad():
                        if 'llama' in args.model.lower():
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                        elif 'opt' in args.model.lower():
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                for h in handles:
                    h.remove()

                for name in gptq_subset:
                    print('Quantizing ...')
                    gptq[name].fasterquant(
                        percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order,
                        static_groups=args.static_groups
                    )
                    subset[name].weight.data = gptq[name].layer.weight.data
                    gptq[name].free()

        # elif args.quant_type == 'awq':
        #     for name in subset:
        #         print(f"quantization layer {i} name {name}")
        #         # m.cuda()
        #         subset[name].weight.data = pseudo_quantize_tensor(
        #             subset[name].weight.data, n_bit=args.wbits, **q_config
        #         )
        #         # m.cpu()
        layer = layer.cpu().to(dtype=dtype)
        use_new_forward(layer, recurse=True)
        layers[i] = layer
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if args.quant_type == 'awq':
        q_config = {
            "zero_point": not args.no_zero_point,  # by default True
            "q_group_size": args.q_group_size,  # whether to use group quantization
        }
        print("Quantization config:", q_config)

        # 如果指定了加载预先计算好的 AWQ 结果
        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            # 加载 scale 和 clip 参数
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model.model, awq_results)

        for i in range(len(layers)):
            layer = layers[i].to(device)
            use_old_forward(layer, recurse=True)
            subset = find_layers(layer)

            for name in subset:
                print(f"quantization layer {i} name {name}")
                subset[name].weight.data = pseudo_quantize_tensor(
                    subset[name].weight.data, n_bit=args.wbits, **q_config
                )

            layer = layer.cpu().to(dtype=dtype)
            use_new_forward(layer, recurse=True)
            layers[i] = layer
            del layer
            torch.cuda.empty_cache()

    prune_time_cost = time.time() - prune_start
    print(f'Prune time cost: {prune_time_cost / 60:.2f} minutes')
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # return model
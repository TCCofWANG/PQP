import gc
import time
import os
import torch
from timm.utils import NativeScaler
from utils.options import args
import contextlib
from tqdm import trange
from optimizers import Prodigy
from models.sparse_layers import SparseLinear, WrappedGPT, SparseGPT
from utils.gptq import GPTQ
from utils.quant import Quant, quant
from utils.data import get_loaders
from utils.tools import *

inference_context = contextlib.nullcontext() if args.use_fp32 else torch.cuda.amp.autocast()

USE_WANDB = False
if USE_WANDB:
    import wandb
else:
    class Wandb:
        def __init__(self): pass

        def login(self): pass

        def init(self, **kwargs): pass

        def finish(self): pass

        def log(self, log_info):
            print(f"wandb: {log_info}")
    wandb = Wandb()

def loss_func(l2_loss_1, l2_loss_2, sparsity):
    loss = args.l2_alpha * l2_loss_1 +  args.l2_beta * l2_loss_2 + args.sparsity_beta * ((sparsity - args.sparsity) / args.sparsity) ** 2
    return loss

def wrapped_forward(layer, inps, attention_mask=None, position_ids=None, repeat_size=1):
    attention_mask = attention_mask.repeat(repeat_size, 1, 1, 1) if attention_mask is not None else attention_mask
    position_ids = position_ids.repeat(repeat_size, 1) if position_ids is not None else position_ids
    if 'llama' in args.model.lower():
        return layer(inps, attention_mask=attention_mask, position_ids=position_ids)
    elif 'opt' in args.model.lower():
        return layer(inps, attention_mask=attention_mask)

def val_epoch(layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs, refer_dense=False,):
    refer_outs = dense_outs if refer_dense else outs
    with torch.no_grad():
        l2_loss1_list, l2_loss2_list, loss_list, dense_l2_loss_list = [], [], [], []
        sparsity = float(get_sparsity(sparse_layers))
        if args.norm_all:
            l2_scaler = torch.norm(refer_outs.type(torch.float32).reshape((-1, refer_outs.shape[-1])).t(), p=2, dim=1)

        for begin_idx in range(0, args.nsamples, args.prune_batch_size):
            # l2_loss_2 = 0
            end_idx = min(args.nsamples, begin_idx + args.prune_batch_size)
            with inference_context:
                if not args.norm_all:
                    l2_scaler = torch.norm(refer_outs[begin_idx: end_idx, ].type(torch.float32).reshape((-1, refer_outs[begin_idx: end_idx, ].shape[-1])).t(), p=2, dim=1).detach()
                # ***********************************#
                pruned_outs[begin_idx: end_idx, ] = wrapped_forward(layer, inps[begin_idx: end_idx, ], attention_mask, position_ids, end_idx - begin_idx)[0]
                if args.quant_error_loss:
                    get_quantization_error_true(sparse_layers)
                    quantization_error_out = wrapped_forward(layer, inps[begin_idx: end_idx, ], attention_mask, position_ids, end_idx - begin_idx)[0]
                    get_quantization_error_false(sparse_layers)

                    l2_loss1 = (((refer_outs[begin_idx: end_idx, ] - pruned_outs[begin_idx: end_idx, ]) / l2_scaler) ** 2).sum() / refer_outs[begin_idx: end_idx, ].shape[-1]
                    l2_loss2 = ((quantization_error_out / l2_scaler) ** 2).sum() / refer_outs[begin_idx: end_idx, ].shape[-1]
                else:
                    l2_loss1 = (((refer_outs[begin_idx: end_idx, ] - pruned_outs[begin_idx: end_idx, ]) / l2_scaler) ** 2).sum() / refer_outs[begin_idx: end_idx, ].shape[-1]
                    l2_loss2 = torch.tensor(0.0, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                loss = loss_func(l2_loss1, l2_loss2, sparsity)
                if not args.no_dense_loss:
                    dense_l2_loss = ((dense_outs[begin_idx: end_idx, ] - pruned_outs[begin_idx: end_idx, ]) ** 2).sum() / pruned_outs[begin_idx: end_idx, ].numel()
                    dense_l2_loss_list.append(dense_l2_loss.item())
            loss_list.append(float(loss))
            l2_loss1_list.append((l2_loss1).item())
            l2_loss2_list.append((l2_loss2).item())
    val_loss = sum(loss_list) / len(loss_list)
    val_l2_loss1 = sum(l2_loss1_list) / len(l2_loss1_list)
    val_l2_loss2 = sum(l2_loss2_list) / len(l2_loss2_list)
    return sparsity, val_loss, val_l2_loss1, val_l2_loss2, dense_l2_loss_list

def train_epoch(layer, sparse_layers, attention_mask, position_ids, inps, refer_outs, optimizer, loss_scaler, train_params):
    l2_loss1_list, l2_loss2_list, loss_list = [], [], []
    if args.norm_all:
        l2_scaler = torch.norm(refer_outs.type(torch.float32).reshape((-1, refer_outs.shape[-1])).t(), p=2,dim=1).detach()
    for begin_idx in range(0, args.nsamples, args.prune_batch_size):
        end_idx = min(args.nsamples, begin_idx + args.prune_batch_size)
        with inference_context:
            if not args.norm_all:
                l2_scaler = torch.norm(refer_outs[begin_idx: end_idx, ].type(torch.float32).reshape((-1, refer_outs[begin_idx: end_idx, ].shape[-1])).t(), p=2, dim=1).detach()
            pruned_out = wrapped_forward(layer, inps[begin_idx: end_idx, ], attention_mask, position_ids, end_idx - begin_idx)[0]
            if args.quant_error_loss:
                get_quantization_error_true(sparse_layers)
                quantization_error_out = wrapped_forward(layer, inps[begin_idx: end_idx, ], attention_mask, position_ids, end_idx - begin_idx)[0]
                get_quantization_error_false(sparse_layers)

                sparsity = get_sparsity(sparse_layers)
                l2_loss1 = (((refer_outs[begin_idx: end_idx, ] - pruned_out) / l2_scaler) ** 2).sum() / refer_outs[begin_idx: end_idx, ].shape[-1]
                l2_loss2 = ((quantization_error_out / l2_scaler) ** 2).sum() / refer_outs[begin_idx: end_idx, ].shape[-1]
            else:
                sparsity = get_sparsity(sparse_layers)
                l2_loss1 = (((refer_outs[begin_idx: end_idx, ] - pruned_out) / l2_scaler) ** 2).sum() / refer_outs[begin_idx: end_idx, ].shape[-1]
                l2_loss2 = torch.tensor(0.0, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            loss = loss_func(l2_loss1, l2_loss2, sparsity)
            loss_list.append(loss.item())
            l2_loss1_list.append((l2_loss1).item())
            l2_loss2_list.append((l2_loss2).item())
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, parameters=train_params, clip_grad=args.clip_grad, clip_mode=args.clip_mode)
        torch.cuda.empty_cache()

    train_loss = sum(loss_list) / len(loss_list)
    train_l2_loss1 = sum(l2_loss1_list) / len(l2_loss1_list)
    train_l2_loss2 = sum(l2_loss2_list) / len(l2_loss2_list)

    return train_loss, train_l2_loss1, train_l2_loss2

def grad_prune(layer_index, layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs):
    print(f"Grad prune layer {layer_index}")
    sparsity_params = get_sparsity_params(sparse_layers)
    lora_params = get_lora_params(sparse_layers)
    if len(lora_params) > 0:
        param_lr = args.prodigy_lr if not args.normal_opt else 1e-3 if args.normal_default else args.normal_opt_lr
        compress_params = [
            {'params': sparsity_params, 'lr': param_lr},
            {'params': lora_params, 'lr': param_lr},
        ]
        train_params = sparsity_params + lora_params
    else:
        compress_params = train_params = sparsity_params
    loss_scaler = FakeScaler() if args.no_scaler else NativeScaler()

    if args.normal_opt:
        if args.normal_default:
            optimizer = torch.optim.AdamW(compress_params)
        else:
            optimizer = torch.optim.AdamW(compress_params, lr=args.normal_opt_lr, weight_decay=0)
    else:
        optimizer = Prodigy(compress_params, args.prodigy_lr,
                            weight_decay=args.weight_decay,
                            decouple=not args.no_decouple,
                            use_bias_correction=args.use_bias_correction,
                            safeguard_warmup=args.safeguard_warmup,
                            d_coef=args.d_coef
                            )

    if args.use_cos_sche:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # learn sparsity epochs
    refer_outs = dense_outs if args.prune_dense else outs
    for epoch in range(args.epochs):
        # train epoch
        train_loss, train_l2_loss1, train_l2_loss2 = train_epoch(layer, sparse_layers, attention_mask, position_ids, inps, refer_outs, optimizer, loss_scaler, train_params)
        if args.use_cos_sche:
            lr_scheduler.step(epoch)
        torch.cuda.empty_cache()

        # val epoch
        sparsity, val_loss, val_l2_loss1, val_l2_loss2, dense_l2_loss_list = val_epoch(layer, sparse_layers, attention_mask,
                                                                        position_ids, inps, outs, pruned_outs,
                                                                        dense_outs, args.prune_dense)

        wandb_log = {
            f'layer_{layer_index}-train_loss': train_loss,
            f'layer_{layer_index}-train_l2_loss1': train_l2_loss1,
            f'layer_{layer_index}-train_l2_loss2': train_l2_loss2,
            f'layer_{layer_index}-sparsity': sparsity,
            f'layer_{layer_index}-val_loss': val_loss,
            f'layer_{layer_index}-val_l2_loss1': val_l2_loss1,
            f'layer_{layer_index}-val_l2_loss2': val_l2_loss2,
        }
        if not args.no_dense_loss:
            dense_val_l2_loss = sum(dense_l2_loss_list) / len(dense_l2_loss_list)
            wandb_log[f'layer_{layer_index}-dense_val_l2_loss'] = dense_val_l2_loss
        for layer_name in sparse_layers:
            sparse_layer = sparse_layers[layer_name]
            wandb_log[f"layer_{layer_index}-{layer_name}_sparsity"] = float(sparse_layer.sparsity)
        wandb.log(wandb_log)

    return wandb_log, sparsity

def fixed_prune(layer_index, layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs):
    print(f"Fixed prune layer {layer_index}")
    sparsity, val_loss, val_l2_loss1, val_l2_loss2, dense_l2_loss_list = val_epoch(layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs, args.prune_dense)
    wandb_log = {
        f'layer_{layer_index}-val_loss': val_loss,
        f'layer_{layer_index}-val_l2_loss': val_l2_loss1,
        f'layer_{layer_index}-sparsity': args.sparsity,
    }
    if not args.no_dense_loss:
        dense_val_l2_loss = val_l2_loss1 if args.prune_dense else sum(dense_l2_loss_list) / len(dense_l2_loss_list)
        wandb_log[f'layer_{layer_index}-dense_val_l2_loss'] = dense_val_l2_loss
    wandb.log(wandb_log)

    return wandb_log, sparsity

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def compress_model(args, model, dataloader):
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

    def add_batch(layer_name):
        def tmp(_, inp, out):
            sparse_layers[layer_name].add_batch(inp[0].data, out.data)
        return tmp

    print('Starting ...')
    prune_start = time.time()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if 'llama' in args.model.lower():
        layers = model.model.model.layers
        model.model.model.embed_tokens = model.model.model.embed_tokens.to(dev)
    elif "opt" in args.model.lower():
        layers = model.model.model.decoder.layers
        model.model.model.decoder.embed_tokens = model.model.model.decoder.embed_tokens.to(dev)
        model.model.model.decoder.embed_positions = model.model.model.decoder.embed_positions.to(dev)

    if args.gradient_path:
        with open(args.gradient_path, 'rb') as file:
            gradients = torch.load(args.gradient_path, map_location=torch.device('cpu'))

    dtype = next(iter(model.model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), device=dev, dtype=dtype
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])
    for i in range(args.nsamples):
        try:
            batch = dataloader[i]
            model.model(batch[0].to(dev))
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

    attention_mask = cache['attention_mask']
    if args.use_fp32:
        inps = inps.float()
        attention_mask = attention_mask.float()
        dtype = torch.float32
    torch.cuda.empty_cache()

    pruned_outs = torch.zeros_like(inps)
    if args.prune_dense or (not args.no_dense_loss):
        dense_inps = inps.clone()
        dense_outs = torch.zeros_like(inps)
    else:
        dense_outs = None
    outs = None if args.prune_dense else torch.zeros_like(inps)

    model_prune_log, model_sparsity = [], []
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        use_old_forward(layer, recurse=True)
        if args.use_fp32:
            layer = layer.float()

        layer.self_attn.q_proj = SparseLinear(layer.self_attn.q_proj, args.metric_type, args.wise_dim)
        layer.self_attn.k_proj = SparseLinear(layer.self_attn.k_proj, args.metric_type, args.wise_dim)
        layer.self_attn.v_proj = SparseLinear(layer.self_attn.v_proj, args.metric_type, args.wise_dim)
        if 'llama' in args.model.lower():
            layer.self_attn.o_proj = SparseLinear(layer.self_attn.o_proj, args.metric_type, args.wise_dim)
            layer.mlp.gate_proj = SparseLinear(layer.mlp.gate_proj, args.metric_type, args.wise_dim)
            layer.mlp.up_proj = SparseLinear(layer.mlp.up_proj, args.metric_type, args.wise_dim)
            layer.mlp.down_proj = SparseLinear(layer.mlp.down_proj, args.metric_type, args.wise_dim)
        elif 'opt' in args.model.lower():
            layer.self_attn.out_proj = SparseLinear(layer.self_attn.out_proj, args.metric_type, args.wise_dim)
            layer.fc1 = SparseLinear(layer.fc1, args.metric_type, args.wise_dim)
            layer.fc2 = SparseLinear(layer.fc2, args.metric_type, args.wise_dim)

        handles = []
        sparse_layers = find_layers(layer, layers=[SparseLinear])
        for layer_name in sparse_layers:
            sparse_layer = sparse_layers[layer_name]
            handles.append(sparse_layer.register_forward_hook(add_batch(layer_name)))
        with inference_context:
            refer_outs = pruned_outs if outs is None else outs
            for begin_idx in trange(0, args.nsamples, args.prune_batch_size, desc="calc refer outs before quantization", leave=False):
                end_idx = min(args.nsamples, begin_idx + args.prune_batch_size)
                refer_outs[begin_idx: end_idx, ] = wrapped_forward(layer, inps[begin_idx: end_idx, ], attention_mask, position_ids, end_idx - begin_idx)[0]
                torch.cuda.empty_cache()
        for h in handles:
            h.remove()

        if args.prune_dense or (not args.no_dense_loss):
            with inference_context:
                for begin_idx in range(0, args.nsamples, args.prune_batch_size):
                    end_idx = min(args.nsamples, begin_idx + args.prune_batch_size)
                    dense_outs[begin_idx: end_idx, ] = wrapped_forward(layer, dense_inps[begin_idx: end_idx, ], attention_mask, position_ids, end_idx - begin_idx)[0]
                    torch.cuda.empty_cache()

        Init_W = {}
        if args.wbits < 16:
            if args.quant_type == 'rtn':
                print(f"---------------- RTN Layer {i} of {len(layers)} ----------------")
                for name in sparse_layers:
                    indexed_name = f"{name}_layer_{i}"
                    print(f'RTN quantization {name} layer {i}')
                    Init_W[indexed_name] = sparse_layers[name].weight.data.clone()
                    quantizer = Quant()
                    quantizer.configure(args.wbits, perchannel=True, sym=False)

                    W = Init_W[indexed_name]
                    quantizer.find_params(W, weight=True)
                    Q = quant(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                        next(iter(layer.parameters())).dtype).clone()
                    sparse_layers[name].quantization_error = (W - Q).to(torch.float32)
                    sparse_layers[name].weight.data = Q

            elif args.quant_type == 'gptq' and args.metric_type != 'sparsegpt':
                full = sparse_layers
                print(f"---------------- GPTQ Layer {i} of {len(layers)} ----------------")
                if args.true_sequential:
                    sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'],
                                  ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
                else:
                    sequential = [list(full.keys())]

                for names in sequential:
                    gptq_layers = {n: full[n] for n in names}
                    gptq = {}
                    for name in gptq_layers:
                        gptq[name] = GPTQ(gptq_layers[name])
                        gptq[name].quantizer = Quant()
                        gptq[name].quantizer.configure(
                            args.wbits, perchannel=True, sym=args.sym, mse=False
                        )

                    def gptq_add_batch(name):
                        def tmp(_, inp, out):
                            gptq[name].add_batch(inp[0].data)
                        return tmp

                    handles = []
                    for name in gptq_layers:
                        handles.append(gptq_layers[name].register_forward_hook(gptq_add_batch(name)))
                    for j in trange(args.nsamples, desc="calc outs before quantization", leave=False):
                        with torch.no_grad():
                            if 'llama' in args.model.lower():
                                pruned_outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                            elif 'opt' in args.model.lower():
                                pruned_outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    for h in handles:
                        h.remove()

                    for name in gptq_layers:
                        indexed_name = f"{name}_layer_{i}"
                        Init_W[indexed_name] = sparse_layers[name].layer.weight.data.clone()
                        W = Init_W[indexed_name]
                        # print(i, name)
                        print('Quantizing ...')
                        gptq[name].fasterquant(
                            percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order,
                            static_groups=args.static_groups
                        )
                        Q = gptq[name].layer.weight.data.clone()
                        sparse_layers[name].weight.data = gptq[name].layer.weight.data
                        sparse_layers[name].quantization_error = (W - Q).to(torch.float32)
                        gptq[name].free()

        if args.metric_type == 'wanda':
            for name in sparse_layers:
                X = sparse_layers[name].scaler_row.reshape((1, -1))
                W_metric = torch.abs(sparse_layers[name].weight.data) * torch.sqrt(X)
                sparse_layers[name].W_metric = W_metric
                # W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                # sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity)]
                # W_mask.scatter_(1, indices, True)
                # sparse_layers[name].weight.data[W_mask] = 0  ## set weights to zero
        elif args.metric_type == 'pqp':
            for name in sparse_layers:
                indexed_name = f'{name}_layer_{i}'
                X = sparse_layers[name].scaler_row.reshape((1, -1))
                W = Init_W[indexed_name]
                # W_wanda = torch.abs(W) * torch.sqrt(X)
                W_ria = ((torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W),dim=1).reshape(-1,1)) * (torch.sqrt(X)) ** 0.5)
                W_metric = torch.abs(sparse_layers[name].weight.data) * mms(W_ria)
                sparse_layers[name].W_metric = W_metric
        elif args.metric_type == 'pruner-zero':
            for name in sparse_layers:
                indexed_name = f'{name}_layer_{i}'
                G = gradients[indexed_name]
                W_metric = mul(mul(sparse_layers[name].weight.data.to(dtype=torch.float32),
                                   sparse_layers[name].weight.data.to(dtype=torch.float32)),
                               mms(abs(G.to(device=sparse_layers[name].device, dtype=torch.float32))))
                sparse_layers[name].W_metric = W_metric
        elif args.metric_type == 'pqp-metric-G':
            for name in sparse_layers:
                indexed_name = f'{name}_layer_{i}'
                G = gradients[indexed_name]
                W = Init_W[indexed_name]
                W_prune_zero = pow(W.to(dtype=torch.float32)) * mms(abs(G.to(device=W.device, dtype=torch.float32)))
                W_metric = abs(sparse_layers[name].weight.data.to(dtype=torch.float32)) * mms(W_prune_zero)
                sparse_layers[name].W_metric = W_metric
        elif args.metric_type == 'magnitude':
            for name in sparse_layers:
                W = sparse_layers[name].weight.data
                W_metric = torch.abs(W)
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity)].cpu()
                W_mask = (W_metric <= thresh)
                W[W_mask] = 0
        elif args.metric_type == 'ria':
            for name in sparse_layers:
                X = sparse_layers[name].scaler_row.reshape((1, -1))
                W_metric = ((torch.abs(sparse_layers[name].weight.data)/torch.sum(torch.abs(sparse_layers[name].weight.data), dim=0)
                                + torch.abs(sparse_layers[name].weight.data)/torch.sum(torch.abs(sparse_layers[name].weight.data), dim=1).reshape(-1, 1))
                                * (torch.sqrt(X)**0.5))
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity)]
                W_mask.scatter_(1, indices, True)
                sparse_layers[name].weight.data[W_mask] = 0  ## set weights to zero
        elif args.metric_type == 'pqp-metric':
            for name in sparse_layers:
                indexed_name = f'{name}_layer_{i}'
                X = sparse_layers[name].scaler_row.reshape((1, -1))
                W = Init_W[indexed_name]
                W_ria = ((torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W),dim=1).reshape(-1,1)) * (torch.sqrt(X)) ** 0.5)
                W_metric = torch.abs(sparse_layers[name].weight.data) * mms(W_ria)
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity)]
                W_mask.scatter_(1, indices, True)
                sparse_layers[name].weight.data[W_mask] = 0  ## set weights to zero
        elif args.metric_type == 'sparsegpt':
            print(f"---------------- GPTQ Layer {i} of {len(layers)} ----------------")
            full = sparse_layers
            if args.true_sequential:
                sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'],
                              ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
            else:
                sequential = [list(full.keys())]

            for names in sequential:
                gpts_sparse_layers = {n: full[n] for n in names}

                gpts = {}
                for name in gpts_sparse_layers:
                    gpts[name] = SparseGPT(gpts_sparse_layers[name])
                    if args.wbits < 16:
                        if args.quant_type == "gptq":
                            gpts[name].quantizer = Quant()
                            gpts[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

                def gpts_add_batch(name):
                    def tmp(_, inp, out):
                        gpts[name].add_batch(inp[0].data, out.data)
                    return tmp

                handles = []
                for name in gpts:
                    handles.append(gpts_sparse_layers[name].register_forward_hook(gpts_add_batch(name)))

                for j in trange(args.nsamples, desc="calc outs before pruning", leave=False):
                    with torch.no_grad():
                        if 'llama' in args.model.lower():
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                        elif 'opt' in args.model.lower():
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                for h in handles:
                    h.remove()

                for name in gpts:
                    print('Quantization and Pruning ...')
                    gpts[name].faster_quant_prune(args.sparsity, prune_n=0, prune_m=0, percdamp=args.percdamp, blocksize=128)
                    sparse_layers[name].weight.data = gpts[name].layer.weight.data
                    gpts[name].free()

        print(f"---------------- Prune Layer {i} of {len(layers)} ----------------")
        prune_func = grad_prune
        if args.fix_layers:
            fix_layers = list(sparse_layers.keys()) if args.fix_layers == 'all' else args.fix_layers.split(',')
            prune_func = fixed_prune if args.fix_layers == 'all' else grad_prune
            for layer_name in fix_layers:
                sparse_layers[layer_name].sparsity = args.sparsity

        torch.set_grad_enabled(True)
        init_learn_sparsity(sparse_layers, args.sparsity_step, blocksize=args.blocksize, sigmoid_smooth=not args.no_sigmoid_smooth, lora_rank=args.lora_rank)
        layer_prune_log, layer_sparsity = prune_func(i, layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs)
        torch.set_grad_enabled(False)
        finish_learn_sparsity(sparse_layers)  # 得到一个层中每个模块的稀疏度 下次进入下一层
        model_prune_log.append(layer_prune_log)
        model_sparsity.append(layer_sparsity)

        layer.self_attn.q_proj = layer.self_attn.q_proj.layer
        layer.self_attn.k_proj = layer.self_attn.k_proj.layer
        layer.self_attn.v_proj = layer.self_attn.v_proj.layer
        if 'llama' in args.model.lower():
            layer.self_attn.o_proj = layer.self_attn.o_proj.layer
            layer.mlp.gate_proj = layer.mlp.gate_proj.layer
            layer.mlp.up_proj = layer.mlp.up_proj.layer
            layer.mlp.down_proj = layer.mlp.down_proj.layer
        elif 'opt' in args.model.lower():
            layer.self_attn.out_proj = layer.self_attn.out_proj.layer
            layer.fc1 = layer.fc1.layer
            layer.fc2 = layer.fc2.layer

        layer = layer.cpu().to(dtype=dtype)
        use_new_forward(layer, recurse=True)
        layers[i] = layer
        del layer
        del sparse_layers
        if args.quant_type == 'gptq' and args.metric_type != 'sparsegpt':
            del gptq
        if args.metric_type == 'sparsegpt':
            del gpts
        del Init_W
        gc.collect()
        torch.cuda.empty_cache()

        inps, pruned_outs = pruned_outs, inps
        if args.prune_dense or (not args.no_dense_loss):
            dense_inps, dense_outs = dense_outs, dense_inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    prune_time_cost = time.time() - prune_start
    print(f'Prune time cost: {prune_time_cost/60:.2f} minutes')
    model_sparsity = sum(model_sparsity) / len(model_sparsity)
    print(f"Model sparsity: {model_sparsity:.6f}")
    return model_prune_log, model_sparsity

def prune_magnitude(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
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

    for i in range(len(layers)):
        layer = layers[i].to(device)
        use_old_forward(layer, recurse=True)
        subset = find_layers(layer)

        if args.wbits < 16:
            if args.quant_type == "rtn":
                for name in subset:
                    print('RTN quantization.....')
                    quantizer = Quant()
                    quantizer.configure(args.wbits, perchannel=True, sym=False)
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quant(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                        next(iter(layer.parameters())).dtype)
            elif args.quant_type == 'gptq':
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
                    for j in trange(args.nsamples, desc="calc outs before quantization", leave=False):
                        with torch.no_grad():
                            if 'llama' in args.model.lower():
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                            elif 'opt' in args.model.lower():
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    for h in handles:
                        h.remove()

                    for name in gptq_subset:
                        # print(i, name)
                        print('Quantizing ...')
                        gptq[name].fasterquant(
                            percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order,
                            static_groups=args.static_groups
                        )
                        subset[name].weight.data = gptq[name].layer.weight.data
                        gptq[name].free()
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity)].cpu()
                W_mask = (W_metric<=thresh)
            subset[name].weight.data[W_mask] = 0

        for j in trange(args.nsamples, desc="calc outs after pruning", leave=False):
            with torch.no_grad():
                if 'llama' in args.model.lower():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif 'opt' in args.model.lower():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layer = layer.cpu().to(dtype=dtype)
        use_new_forward(layer, recurse=True)
        layers[i] = layer
        del layer
        if args.quant_type == 'gptq':
            del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    prune_time_cost = time.time() - prune_start
    print(f'Prune time cost: {prune_time_cost / 60:.2f} minutes')
    model.config.use_cache = use_cache

def prune_sparsegpt(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
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
    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(device)
        use_old_forward(layer, recurse=True)
        subset = find_layers(layer)
        if args.wbits < 16:
            if args.quant_type == "rtn":
                print('RTN quantization.....')
                for name in subset:
                    quantizer = Quant()
                    quantizer.configure(args.wbits, perchannel=True, sym=False)
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quant(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(next(iter(layer.parameters())).dtype)
        full = subset
        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        print(f"---------------- SparseGPT Layer {i} of {len(layers)} ----------------")
        for names in sequential:
            gpts_subset = {n: full[n] for n in names}

            gpts = {}
            for name in gpts_subset:
                gpts[name] = SparseGPT(gpts_subset[name])
                if args.wbits < 16:
                    if args.quant_type == "gptq":
                        gpts[name].quantizer = Quant()
                        gpts[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in gpts_subset:
                handles.append(gpts_subset[name].register_forward_hook(add_batch(name)))

            for j in trange(args.nsamples, desc="calc outs before pruning", leave=False):
                with torch.no_grad():
                    if 'llama' in args.model.lower():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    elif 'opt' in args.model.lower():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in gpts_subset:
                print('Quantization and Pruning ...')
                gpts[name].faster_quant_prune(args.sparsity, prune_n=prune_n, prune_m=prune_m, percdamp=args.percdamp, blocksize=128)
                gpts[name].free()

        for j in trange(args.nsamples, desc="calc outs after pruning", leave=False):
            with torch.no_grad():
                if 'llama' in args.model.lower():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif 'opt' in args.model.lower():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layer = layer.cpu().to(dtype=dtype)
        use_new_forward(layer, recurse=True)
        layers[i] = layer
        del layer
        del gpts
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    prune_time_cost = time.time() - prune_start
    print(f'Prune time cost: {prune_time_cost / 60:.2f} minutes')
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_wanda_ria(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
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

    for i in range(len(layers)):
        layer = layers[i].to(device)
        use_old_forward(layer, recurse=True)
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if 'llama' in args.model.lower():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif 'opt' in args.model.lower():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        if args.wbits < 16:
            if args.quant_type == 'rtn':
                for name in subset:
                    print('RTN quantization.....')
                    # indexed_name = f"{name}_layer_{i}"
                    quantizer = Quant()
                    quantizer.configure(args.wbits, perchannel=True, sym=False)
                    W = subset[name].weight.data
                    # W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quant(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(next(iter(layer.parameters())).dtype)
                    # print("======================== RTN weight =====================")
                    # print(subset[name].weight.data)
            elif args.quant_type == 'gptq':
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
                    for j in trange(args.nsamples, desc="calc outs before quantization", leave=False):
                        with torch.no_grad():
                            if 'llama' in args.model.lower():
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                            elif 'opt' in args.model.lower():
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    for h in handles:
                        h.remove()

                    for name in gptq_subset:
                        print(i, name)
                        print('Quantizing ...')
                        gptq[name].fasterquant(
                            percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order,
                            static_groups=args.static_groups
                        )
                        # quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                        subset[name].weight.data = gptq[name].layer.weight.data
                        gptq[name].free()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            if args.metric_type == 'ria':
                W_metric = ((torch.abs(subset[name].weight.data)/torch.sum(torch.abs(subset[name].weight.data), dim=0)
                            + torch.abs(subset[name].weight.data)/torch.sum(torch.abs(subset[name].weight.data), dim=1).reshape(-1, 1))
                            * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**0.5)
            else:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity)]
                    W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in trange(args.nsamples, desc="calc outs after pruning", leave=False):
            with torch.no_grad():
                if 'llama' in args.model.lower():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif 'opt' in args.model.lower():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layer = layer.cpu().to(dtype=dtype)
        use_new_forward(layer, recurse=True)
        layers[i] = layer
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        if args.quant_type == 'gptq':
            del gptq

    prune_time_cost = time.time() - prune_start
    print(f'Prune time cost: {prune_time_cost / 60:.2f} minutes')
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_pruner_zero(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    prune_start = time.time()
    layers = model.model.model.layers
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.model.model.embed_tokens = model.model.model.embed_tokens.to(device)

    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(
            args.gradient_path, map_location=torch.device('cpu'))

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
    model.model.model.embed_tokens = model.model.model.embed_tokens.cpu()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    torch.cuda.empty_cache()

    for i in range(len(layers)):
        layer = layers[i].to(device)
        use_old_forward(layer, recurse=True)
        subset = find_layers(layer)

        if args.wbits < 16:
            if args.quant_type == "rtn":
                for name in subset:
                    print('RTN quantization.....')
                    indexed_name = f"{name}_layer_{i}"
                    quantizer = Quant()
                    quantizer.configure(args.wbits, perchannel=True, sym=False)
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    Q = quant(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                        next(iter(layer.parameters())).dtype)
                    subset[name].weight.data = Q
            elif args.quant_type == 'gptq':
                print(f"---------------- GPTQ Layer {i} of {len(layers)} ----------------")
                full = subset
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
                    for j in trange(args.nsamples, desc="calc outs before quantization", leave=False):
                        with torch.no_grad():
                            if 'llama' in args.model.lower():
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                            elif 'opt' in args.model.lower():
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    for h in handles:
                        h.remove()

                    for name in gptq_subset:
                        # print(i, name)
                        print('Quantizing ...')
                        gptq[name].fasterquant(
                            percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order,
                            static_groups=args.static_groups
                        )
                        # quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                        subset[name].weight.data = gptq[name].layer.weight.data
                        gptq[name].free()

            for name in subset:
                print(f"pruning layer {i} name {name}")
                indexed_name = f"{name}_layer_{i}"
                G = gradients[indexed_name]
                W = subset[name].weight.data
                W_metric = mul(mul(W.to(dtype=torch.float32), W.to(dtype=torch.float32)), mms(G.to(device=W.device, dtype=torch.float32)))

                # W_metric = mul(|mul(|W|,|W|)|,mms(|G|))
                assert W_metric is not None

                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity) > 0.001) and (
                                alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity)]
                        W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in trange(args.nsamples, desc="calc outs after pruning", leave=False):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layer = layer.cpu().to(dtype=dtype)
        use_new_forward(layer, recurse=True)
        layers[i] = layer
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        if args.quant_type == 'gptq':
            del gptq

    prune_time_cost = time.time() - prune_start
    print(f'Prune time cost: {prune_time_cost / 60:.2f} minutes')
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_pqp(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    prune_start = time.time()
    layers = model.model.model.layers
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.model.model.embed_tokens = model.model.model.embed_tokens.to(device)

    if args.gradient_path:
        with open(args.gradient_path, 'rb') as file:
            gradients = torch.load(args.gradient_path, map_location=torch.device('cpu'))

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
    model.model.model.embed_tokens = model.model.model.embed_tokens.cpu()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    torch.cuda.empty_cache()

    for i in range(len(layers)):
        layer = layers[i].to(device)
        use_old_forward(layer, recurse=True)
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        Init_W = {}
        if args.wbits < 16:
            if args.quant_type == "rtn":
                for name in subset:
                    print('RTN quantization.....')
                    indexed_name = f"{name}_layer_{i}"
                    quantizer = Quant()
                    quantizer.configure(args.wbits, perchannel=True, sym=False)
                    Init_W[indexed_name] = subset[name].weight.data.clone()
                    W = Init_W[indexed_name]
                    # W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    Q = quant(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(next(iter(layer.parameters())).dtype)
                    subset[name].weight.data = Q
            elif args.quant_type == 'gptq':
                print(f"---------------- GPTQ Layer {i} of {len(layers)} ----------------")
                full = subset
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
                    for j in trange(args.nsamples, desc="calc outs before quantization", leave=False):
                        with torch.no_grad():
                            if 'llama' in args.model.lower():
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                            elif 'opt' in args.model.lower():
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    for h in handles:
                        h.remove()

                    for name in gptq_subset:
                        # print(i, name)
                        indexed_name = f"{name}_layer_{i}"
                        Init_W[indexed_name] = subset[name].weight.data.clone()
                        print('Quantizing ...')
                        gptq[name].fasterquant(
                            percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order,
                            static_groups=args.static_groups
                        )
                        # quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                        subset[name].weight.data = gptq[name].layer.weight.data
                        gptq[name].free()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            indexed_name = f"{name}_layer_{i}"
            X = wrapped_layers[name].scaler_row.reshape((1, -1))
            if args.metric_type == 'wanda':
                W = Init_W[indexed_name]
                PMW = torch.abs(W.to(dtype=torch.float32) * sqrt(X))
            elif args.metric_type == 'ria':
                W = Init_W[indexed_name]
                PMW = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (sqrt(X))**0.5
            elif args.metric_type == 'pruner-zero':
                G = gradients[indexed_name]
                W = Init_W[indexed_name]
                PMW = mul(mul(W.to(dtype=torch.float32), W.to(dtype=torch.float32)), mms(G.to(device=W.device, dtype=torch.float32)))
            else:
                PMW = None

            W_metric = torch.abs(subset[name].weight.data.to(dtype=torch.float32)) * mms(PMW)

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in trange(args.nsamples, desc="calc outs after pruning", leave=False):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layer = layer.cpu().to(dtype=dtype)
        use_new_forward(layer, recurse=True)
        layers[i] = layer
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        if args.quant_type == 'gptq':
            del gptq

    prune_time_cost = time.time() - prune_start
    print(f'Prune time cost: {prune_time_cost / 60:.2f} minutes')
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
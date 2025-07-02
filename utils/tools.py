import torch
import torch.nn as nn
import math
import fnmatch
from lm_eval import tasks, evaluator
from accelerate import dispatch_model
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from statistics import mean
import time
import numpy as np

def mul(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        x, y = torch.broadcast_tensors(x, y)
        return x * y
    else:
        return x * y

def mms(x):
    # min-max scale to [0, 1]
    if isinstance(x, torch.Tensor):
        return (x - x.min()) / (x.max() - x.min())
    else:
        return (x - min(x)) / (max(x) - min(x))

def sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(torch.abs(x))
    else:
        return math.sqrt(abs(x))

def pow(x):
    if isinstance(x, torch.Tensor):
        return torch.pow(x, 2)
    else:
        return x**2

def abs(x):
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    else:
        return math.fabs(x)

def zsn(x):
    # z-score normalization
    if isinstance(x, torch.Tensor):
        return (x - x.mean()) / x.std()
    else:
        return (x - mean(x)) / x.std()

def print_dict(data, indent=0):
    """Recursively prints dictionary content with indentation."""
    for key, value in data.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key) + ':')
            print_dict(value, indent + 1)
        else:
            print('  ' * indent + f'{key}: {value}')

def compute_avg_acc(results):
    acc_values = [v['acc'] for k, v in results.items() if 'acc' in v]
    return sum(acc_values) / len(acc_values)

def use_old_forward(module: nn.Module, recurse=False):
    if hasattr(module, '_old_forward'):
        module._new_forward = module.forward
        module.forward = module._old_forward
    
    if recurse:
        for child in module.children():
            use_old_forward(child, recurse)


def use_new_forward(module: nn.Module, recurse=False):
    if hasattr(module, '_new_forward'):
        module.forward = module._new_forward
        delattr(module, "_new_forward")
    
    if recurse:
        for child in module.children():
            use_new_forward(child, recurse)


def auto_map_model(model):
    print(f"Check no split modules: {model.model._no_split_modules}")
    max_memory = get_balanced_memory(model.model, dtype=torch.float16, no_split_module_classes=model.model._no_split_modules)
    print(f"Check max memory: {max_memory}")
    model.model.tie_weights()
    print("Model weights tied")
    device_map = infer_auto_device_map(model.model, dtype=torch.float16, max_memory=max_memory, no_split_module_classes=model.model._no_split_modules)
    print(f"Check device map: {device_map}")
    dispatch_model(model.model, device_map)

class FakeScaler:
    def __call__(self, loss, optimizer, parameters=None, clip_grad=None, clip_mode=None):
        loss.backward()
        optimizer.step()


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def init_learn_sparsity(sparse_layers, sparsity_step=0.01, prune_n=0, prune_m=0, blocksize=-1, sigmoid_smooth=False, lora_rank=-1, lora_alpha=1):
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        sparse_layer.init_learn_sparsity(sparsity_step, prune_n, prune_m, blocksize, sigmoid_smooth, lora_rank, lora_alpha)

def finish_learn_sparsity(sparse_layers):
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        sparse_layer.finish_learn_sparsity()

def get_sparsity(sparse_layers):
    total_param = sum([sparse_layers[layer_name].param_num for layer_name in sparse_layers])
    sparsity = 0
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        sparsity += sparse_layer.sparsity * (sparse_layer.param_num / total_param)
    return sparsity

def get_quantization_error_true(sparse_layers):
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        sparse_layer.use_error = True

def get_quantization_error_false(sparse_layers):
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        sparse_layer.use_error = False

def get_quantization_error(sparse_layers):
    var = 0
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        var += sparse_layer.get_quantization_error_var()
    return var

def get_sparsity_params(sparse_layers):
    params = []
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        if sparse_layer.sparsity_probabilities is not None:
            layer_sparsity_params = sparse_layer.sparsity_probabilities
            if type(layer_sparsity_params) is list:
                params.extend(layer_sparsity_params)
            else:
                params.append(layer_sparsity_params)
    return params

def get_lora_params(sparse_layers):
    params = []
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        if sparse_layer.use_lora:
            params.append(sparse_layer.lora_A)
            params.append(sparse_layer.lora_B)
    return params

def eval_ppl(model, testenc, batch_size=1, device=torch.device("cuda:0")):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    neg_log_likelihoods = []
    for i in range(0, nsamples, batch_size):
        j = min(i + batch_size, nsamples)
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model._model_call(inputs)
        # print(lm_logits)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j - i)
        neg_log_likelihoods.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(neg_log_likelihoods).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()

    return ppl.item()

def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

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

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        # if args.nearest:
        #     subset = find_layers(layer)
        #     for name in subset:
        #         quantizer = quant.Quantizer()
        #         quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
        #         W = subset[name].weight.data
        #         quantizer.find_params(W, weight=True)
        #         subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        print(lm_logits)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def benchmark(model, input_ids, check=False, device='cuda'):
    input_ids = input_ids.to(device)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=device)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(device), input_ids[:, (i + 1)].to(device)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"],
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=/public/lzy/llm_weights"
    limit = None
    if "70b" in model_name or "65b" in model_name or "30b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=/public/lzy/llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        max_batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens
    )
    print("zero_shot evaluation results")
    print(evaluator.make_table(results))
    return results



import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from importlib.metadata import version
from transformers import AdamW
from datasets import load_dataset, load_from_disk
import torch.nn as nn 
from tqdm import tqdm
import argparse
import os
# from utils.sparsegpt import *
from models.llama import LLaMA
from models.llama_seq import LLaMA_Seq

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())
USE_LLaMA_SEQ = torch.cuda.device_count() == 1

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    traindata = load_from_disk("/public/lzy/dataset/wikitext-2-raw-v1/train/")
    testdata = load_from_disk("/public/lzy/dataset/wikitext-2-raw-v1/test/")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        # tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    print("printing gpu allocation for all the layers")
    print(model.hf_device_map)
    model.seqlen = 2048
    return model

def get_model(model_name, batch_size=1):
    def skip(*args, **kwargs):
        pass
    nn.init.kaiming_uniform_ = skip
    nn.init.uniform_ = skip
    nn.init.normal_ = skip

    if 'llama' in model_name.lower() or 'Llama' in model_name.lower():
        if USE_LLaMA_SEQ:
            model = LLaMA_Seq(model_name, batch_size=batch_size)
        else:
            model = LLaMA(model_name, batch_size=batch_size)
    else:
        raise NotImplementedError(f"Invalid model name: {model_name}")
    return model

class gradient_computation:
    def __init__(self, model, scale):
        self.model = model
        self.gradients_l1 = dict()
        self.gradients_l2 = dict()
        self.nsample = 0
        self.scale = scale
        self.device = torch.device("cpu") 
        self.gradients_init()

    def gradients_init(self):
        layers = self.model.model.layers
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in tqdm(range(len(layers)), desc=f"initializing the gradient list ...."):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                self.gradients_l1[indexed_name] = torch.zeros_like(subset[name].weight, dtype=torch.float16, device=self.device)
                self.gradients_l2[indexed_name] = torch.zeros_like(subset[name].weight, dtype=torch.float32, device=self.device)
    
    def update_gradient(self, model, nsample):
        assert nsample - self.nsample == 1, "number of samples must be incremented by 1"
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc=f"updating the gradient of sample no: {self.nsample}"):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                if subset[name].weight.grad is None:
                    print(f"Error: {name} has none gradient")
                if subset[name].weight.grad is not None:
                    assert subset[name].weight.requires_grad == True, f"Required grad must be true ( {name}: {subset[name].weight.requires_grad})"
                    grad = subset[name].weight.grad.detach().clone().to(dtype=torch.float32)  # Cast to float32
                    all_zero = (torch.abs(grad)==0).all()
                    assert int(all_zero) == 0, f"all the elements in the tensor are zero.: {all_zero}"
                    assert self.gradients_l1[indexed_name].shape == grad.shape, "shape mismatch"
                    self.gradients_l1[indexed_name] = self.gradients_l1[indexed_name] + torch.abs(grad*self.scale).to(device=self.device).to(dtype=torch.float16)
                    self.gradients_l2[indexed_name] = self.gradients_l2[indexed_name] + torch.abs((grad*self.scale)**2).to(device=self.device)
        self.nsample = nsample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', type=int, default=128, help='no of samples used')
    parser.add_argument('--scale', type=int, default=100, help='no of samples used')
    parser.add_argument('--llama_version', type=int, default=1, help='llama version used')
    parser.add_argument('--model', type=str, help='model to used') ## change
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='batch size of model evaluation'
    )
    parser.add_argument('--seed', type=int, default=0, help='seed used')
    args = parser.parse_args()
    print(f"Obtaining gradients for no of samples {args.nsamples}, scale {args.scale}")
    
    model_args = args.model
    # if args.llama_version == 2:
    model = get_model(args.model, args.batch_size)
    torch.set_grad_enabled(True)
    tokenizer = AutoTokenizer.from_pretrained(model_args, use_fast=False)
    model = model.model
    # else:
    #     cache_dir_args = "/home/user0/public2/lzy/llm_weights"
    #     model = get_llm(args.model, cache_dir_args)
    if args.llama_version == 2:
        tokenizer = AutoTokenizer.from_pretrained(model_args, use_fast=False)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_args, use_fast=False) ## change

    layers = model.model.layers

    # if "model.embed_tokens" in model.hf_device_map:
    #     device = model.hf_device_map["model.embed_tokens"]
    # else:
    #     device = model.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading calibdation data")
    nsamples=args.nsamples
    seed=args.seed
    dataloader, _ = get_loaders("wikitext2",nsamples=nsamples,seed=seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")

    # print("quantizition model")
    # quant_model(model, dataloader)
    # print("quantizition model complete")

    optimizer = AdamW(model.parameters(), lr=0.01, eps=0.01)
    optimizer.zero_grad()
    grad_up = gradient_computation(model, args.scale)
    nsample = 0
    model.train()
    for input_ids, labels in tqdm(dataloader):
        nsample+=1
        print("making gradient computation on sample: ", nsample)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        grad_up.update_gradient(model, nsample)
        optimizer.zero_grad()
    print("Done")
    gradients_l2 = grad_up.gradients_l2

    for name in gradients_l2:
        grad_sqrt = torch.sqrt(gradients_l2[name])
        gradients_l2[name] = grad_sqrt.to(dtype=torch.float16)
    model_name = os.path.basename(args.model)
    if not os.path.exists(f'./gradients/llama{args.llama_version}'):
        os.makedirs(f'./gradients/llama{args.llama_version}')
    with open(f'./gradients/llama{args.llama_version}/gradients_aggregrate_norm_l2_model_{model_name}_{args.nsamples}_{args.seed}.pth', 'wb') as f:
        torch.save(gradients_l2, f)
    with open(f'./gradients/llama{args.llama_version}/gradients_aggregrate_norm_l1_model_{model_name}_{args.nsamples}_{args.seed}.pth', 'wb') as f:
        torch.save(grad_up.gradients_l1, f)


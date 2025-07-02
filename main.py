import os
import gc
import torch
import torch.nn as nn
import numpy as np
import json
from utils.options import args
from importlib.metadata import version
from models.llama import LLaMA
from models.opt import OPT
from utils.data import get_loaders
from utils.tools import eval_ppl, auto_map_model, eval_zero_shot, print_dict, compute_avg_acc
from utils.prune import wandb, compress_model, prune_magnitude, prune_wanda_ria, prune_sparsegpt, prune_pruner_zero, prune_pqp

# Setting seeds for reproducibility
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

print('cuda', torch.version.cuda)
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_model(model_name, batch_size=1):
    def skip(*args, **kwargs):
        pass
    nn.init.kaiming_uniform_ = skip
    nn.init.uniform_ = skip
    nn.init.normal_ = skip
    if 'llama' in model_name.lower():
        model = LLaMA(args, model_name, batch_size=batch_size)
    elif 'opt' in model_name.lower():
        model = OPT(model_name, batch_size=batch_size)
    else:
        raise NotImplementedError(f"Invalid model name: {model_name}")
    return model


def main(args):
    print("\n============  Loading model... ============")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.batch_size)

    if args.save_path:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    exp_log = f'exp_logs/{args.model_name}/{args.prune_method}'
    os.makedirs(exp_log, exist_ok=True)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    print("\n============ Loading dataset ... ============")
    dataloader, c4_testenc = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=model.tokenizer)
    _, wikitext_testenc = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=model.tokenizer)
    _, ptb_testenc = get_loaders("ptb", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=model.tokenizer)
    ppl_test_sets = ['c4', 'wikitext', 'ptb']
    # ppl_test_sets = ['wikitext']
    gc.collect()
    torch.cuda.empty_cache()

    model_prune_log, eval_result  = [], {}
    avg_acc = 0.0
    model_sparsity = 0.0
    wbits_avg = args.wbits
    if args.sparsity:
        print("\n============ Quantizing and Pruning model... ============")
        if args.prune_method == "magnitude":
            prune_magnitude(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda" or args.prune_method == "ria":
            prune_wanda_ria(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "pruner-zero":
            prune_pruner_zero(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "pqp-metric":
            prune_pqp(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "besa" or args.prune_method == "pqp":
            wandb.login()
            wandb.init(
                project="LLaMA",
                name=args.exp_name,
                config={
                    "model": args.model,
                    "sparsity-step": args.sparsity_step,
                    "epochs": args.epochs,
                    "prune-batch-size": args.prune_batch_size,
                    'l2-alpha': args.l2_alpha,
                    'l2-beta': args.l2_beta,
                    'sparsity-beta': args.sparsity_beta,
                    'fix-layers': args.fix_layers,
                    'prune-dense': args.prune_dense,
                    'dense-loss': not args.no_dense_loss
            })
            model_prune_log, model_sparsity = compress_model(args, model, dataloader)
            wandb.finish()
        else:
            pass
        del dataloader
        torch.cuda.empty_cache()

        auto_map_model(model)

    print(f"max_cuda_mem_quantize_prune: {round(torch.cuda.max_memory_allocated() / 1e9, 2)}")
    print("\n============ Evaluating perplexity... ============")
    c4_ppl = eval_ppl(model, c4_testenc, args.batch_size, device)
    ptb_ppl = eval_ppl(model, ptb_testenc, args.batch_size, device)
    wikitext_ppl = eval_ppl(model, wikitext_testenc, args.batch_size, device)
    for set_name in ppl_test_sets:
        print(f"{set_name} ppl: {eval(f'{set_name}_ppl')}")

    if args.save_path:
        print("\n============ Save post quantization pruning model... ============")
        model.model.save_pretrained(args.save_path)
        model.tokenizer.save_pretrained(args.save_path)

    if args.eval_zero_shot:
        print("\n============ Zero-shot evaluation... ============")
        accelerate = True
        task_list = ["piqa", "boolq", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        # task_list = ["piqa"]
        # task_list = ["rte", "piqa"]
        num_shot = 0
        eval_result = eval_zero_shot(args.model, model.model, model.tokenizer, task_list, num_shot, accelerate)
        avg_acc = compute_avg_acc(eval_result['results'])
        print("Average acc: ", avg_acc)
        eval_result['avg_acc'] = avg_acc


    exp_log = os.path.join(exp_log, f"{args.exp_name}.txt")
    data_to_save = {
        'args': vars(args),  # 使用 vars() 将 argparse.Namespace 对象转换为字典
        'c4_ppl': c4_ppl,
        'ptb_ppl': ptb_ppl,
        'wikitext_ppl': wikitext_ppl,
        'eval_result': eval_result,
        'avg_acc': avg_acc,
        'model_sparsity': model_sparsity,
        'wbits_avg': wbits_avg,
        'model_prune_log': model_prune_log,
    }
    data_to_save_json = json.dumps(data_to_save, indent=4)
    with open(exp_log, 'w') as f:
        f.write(data_to_save_json)



if __name__ == "__main__":
    print(args)
    main(args)

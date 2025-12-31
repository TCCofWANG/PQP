import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='exp_0')
parser.add_argument('--model', type=str, default='/mnt/lustre/share_data/xupeng/llama-7b-hf', help='model to load.')
parser.add_argument('--model-name', type=str, default='llama_7b')
parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity')
parser.add_argument("--prune-method", type=str,
                    choices=["magnitude", "wanda", "sparsegpt", "ria", "pruner-zero", "dense", "besa", "pqp", "pqp-metric", "bawa", "jsq"])
parser.add_argument('--metric-type', type=str, default=None)
parser.add_argument('--quant-type', type=str, default=None)
parser.add_argument('--quant-error-loss', action='store_true')
parser.add_argument("--cache_dir", default="/public/lzy/llm_weights", type=str)
parser.add_argument("--eval_zero_shot", action="store_true")

parser.add_argument('--batch-size', type=int, default=1, help='batch size of model evaluation')
parser.add_argument('--save', type=str, default=None, help='Path to save quant weight.')
parser.add_argument('--load', type=str, default=None, help='Path to load quant weight.')
parser.add_argument('--eval-dense', action='store_true', help='Whether to evaluate the dense model')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--fix-layers', type=str, default=None)
parser.add_argument('--no-dense-loss', action='store_true')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--prune-batch-size', type=int, default=1)
parser.add_argument('--use-fp32', action='store_true')
parser.add_argument('--wise-dim', type=str, default='row')
parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
# Learning parameter settings
parser.add_argument('--blocksize', type=int, default=-1)
parser.add_argument('--sparsity-step', type=float, default=0.01)
parser.add_argument('--lora-rank', type=int, default=-1)
# Loss settings
parser.add_argument('--sparsity-beta', type=float, default=1)
parser.add_argument('--norm-all', action='store_true')
parser.add_argument('--prune-dense', action='store_true')
parser.add_argument('--l2-alpha', type=float, default=1)
parser.add_argument('--l2-beta', type=float, default=1)
parser.add_argument('--no-sigmoid-smooth', action='store_true')
# Scaler (norm, value) and Scheduler
parser.add_argument('--clip-grad', type=float)
parser.add_argument('--clip-mode', type=str, default='norm')
parser.add_argument('--no-scaler', action='store_true')
parser.add_argument('--use-cos-sche', action='store_true')
# Normal Opt settings (AdamW)
parser.add_argument('--normal-opt', action='store_true')
parser.add_argument('--normal-opt-lr', type=float, default=1e-2)
parser.add_argument('--normal-default', action='store_true')
# Prodigy settings
parser.add_argument('--d-coef', type=float, default=1)
parser.add_argument('--prodigy-lr', type=float, default=1)
parser.add_argument('--no-decouple', action='store_true')
parser.add_argument('--use-bias-correction', action='store_true')
parser.add_argument('--safeguard-warmup', action='store_true')
parser.add_argument('--weight-decay', type=float, default=0)
# gradient_path
parser.add_argument("--gradient_path", type=str, default=None, help="Path to save the gradient.")
# gptq quantizition setting
parser.add_argument("--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening.")
parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
parser.add_argument("--wbits", type=int, default=16, help="#bits to use for quantization; use 16 for evaluating base model.")
parser.add_argument("--groupsize", type=int, default=-1, help="How many weight columns (input features) are quantized with the same statistics, default = all of them")
parser.add_argument('--static-groups', action='store_true', help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.')
parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
parser.add_argument("--perchannel", action="store_true", help="fit a unique quantizer to each output dim")
parser.add_argument('--act-order', action='store_true',help='Whether to apply the activation order GPTQ heuristic')
# awq quantizition setting
# parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
# parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])

# parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)

# jsq
# parser.add_argument("--ntrain", "-k", type=int, default=5, help='number of shots')
# parser.add_argument("--ngpu", "-g", type=int, default=8)
# parser.add_argument("--data_dir", "-d", type=str, default="data", required=True, help='dataset location')
# parser.add_argument("--save_dir", "-s", type=str, default="results")
# parser.add_argument("--model", "-m", type=str, required=True)
# parser.add_argument("--path", type=str, required=False, help='model checkpoint location')

# parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
# parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
# parser.add_argument('--seqlen', type=int, default=2048)
# parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
# parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
# parser.add_argument("--cache_dir", default="/mnt/disk1/hg/huggingface/cache", type=str)

# parser.add_argument('--save', type=str, default=None, help='Path to save results.')
# parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
parser.add_argument('--clip_l', type=float, default=0.0)
parser.add_argument('--clip_h', type=float, default=0.01)
parser.add_argument('--abs', action="store_false")
parser.add_argument('--rho', type=float, default=2.1)
# parser.add_argument("--nbits", type=int, default=8)



# # awq quantizition setting
# parser.add_argument("--model_path", type=str, help="path of the hf model")
# parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
# parser.add_argument("--batch_size", type=int, default=1, help="batch size")
# parser.add_argument("--tasks", default=None, type=str)
# parser.add_argument("--output_path", default=None, type=str)
# parser.add_argument("--num_fewshot", type=int, default=0)
#
# # model config
# parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
#
# # quantization config
# parser.add_argument("--w_bit", type=int, default=None)
# parser.add_argument("--q_group_size", type=int, default=-1)
# parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
# parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
#
# # save/load real quantized weights
# parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
# parser.add_argument(
#     "--dump_fake", type=str, default=None, help="save fake-quantized model"
# )
# parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
#
# # apply/save/load awq
# parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
# parser.add_argument(
#     "--dump_awq", type=str, default=None, help="save the awq search results"
# )
# parser.add_argument(
#     "--load_awq", type=str, default=None, help="load the awq search results"
# )
# parser.add_argument(
#     "--vila-15",
#     action="store_true",
#     help="quantizing vila 1.5",
# )
# parser.add_argument(
#     "--vila-20",
#     action="store_true",
#     help="quantizing or smoothing vila 2.0 (NVILA)",
# )
# parser.add_argument(
#     "--smooth_scale",
#     action="store_true",
#     help="generate the act scale of visiontower",
# )
# parser.add_argument(
#     "--media_path",
#     type=str,
#     nargs="+",
#     help="The input video to get act scale for visiontower",
# )
# parser.add_argument(
#     "--act_scale_path",
#     type=str,
#     default=None,
#     help="Path to save act scale",
# )
args = parser.parse_args()
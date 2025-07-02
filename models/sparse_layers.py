import torch
import time
import math
import torch.nn as nn
import mask_gen_cuda
import transformers
from utils.quant import quant

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", reconstruct=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.reconstruct = reconstruct
        if self.reconstruct:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        # if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def faster_quant_prune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, mask=None
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # 如果存在量化
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quant(q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

class SparseLinear(nn.Module):

    def __init__(self, layer, metric_type='wanda', wise_dim='row') -> None:
        super().__init__()
        self.layer = layer
        self.linear_func = nn.functional.linear
        self.register_buffer('weight', layer.weight)
        if layer.bias is not None:
            self.register_buffer('bias', layer.bias)
        else:
            self.bias = None
        self.param_num = self.weight.numel()

        self.nsamples = 0
        self.use_lora = False
        self.learn_sparsity = False
        self.rows = self.weight.data.shape[0]
        self.columns = self.weight.data.shape[1]
        self.device = self.layer.weight.device
        self.quantization_error = torch.zeros_like(self.weight, device=self.device, dtype=torch.float32)
        self.W_metric = None
        self.use_error = False
        self.dtype = None
        self.wise_dim = wise_dim
        assert self.wise_dim in ['row', 'column'], f"Invalid wise dim: {wise_dim}"

        self.metric_type = metric_type
        if metric_type == 'wanda' or metric_type == 'pqp' or self.metric_type == 'ria' or self.metric_type == 'pqp-metric':
            self.scaler_row = torch.zeros((self.columns), device=self.device)
    
    def add_batch(self, inp, out):
        if self.metric_type == 'magnitude':
            return

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        if self.metric_type == 'wanda' or self.metric_type == 'pqp' or self.metric_type == 'ria' or self.metric_type == 'pqp-metric':
            self.scaler_row *= self.nsamples / (self.nsamples+tmp)

        self.nsamples += tmp

        if self.metric_type == 'wanda' or self.metric_type == 'pqp' or self.metric_type == 'ria' or self.metric_type == 'pqp-metric':
            self.scaler_row += torch.norm(inp.float(), p=2, dim=1) ** 2  / self.nsamples

    def get_w_metric(self):
        if self.wise_dim == 'row':
            self.W_metric_sort_indices = torch.sort(self.W_metric, dim=-1, stable=True)[1]
        elif self.wise_dim == 'column':
            self.W_metric_sort_indices = torch.sort(self.W_metric, dim=0, stable=True)[1]
        else:
            raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")

    def init_learn_sparsity(self, sparsity_step=0.01, prune_n=0, prune_m=0, blocksize=-1, sigmoid_smooth=False, lora_rank=-1, lora_alpha=1):
        self.prune_n, self.prune_m = prune_n, prune_m

        if hasattr(self, 'sparsity'):
            self.block_wise = False
            self.learn_sparsity = False
            # self.W_mask = self.get_weight_mask().detach()
            # self.weight.data *= self.W_mask.to(dtype=self.weight.dtype)
            self.finish_learn_sparsity()
            return

        self.get_w_metric()
        torch.cuda.empty_cache()

        self.learn_sparsity = True
        self.block_wise = blocksize != -1
        self.sigmoid_smooth = sigmoid_smooth
        self.sparsity_candidates = torch.arange(1.0, -1 * sparsity_step, -1 * sparsity_step, device=self.device)
        self.sparsity_candidates[-1] = 0.0
        if self.block_wise:
            self.blocksize = blocksize
            if self.wise_dim == 'row':
                assert self.rows % blocksize == 0, "Row blocksize should be fully divided by the number of rows"
                self.blocknum = self.rows // blocksize
            elif self.wise_dim == 'column':
                assert self.columns % blocksize == 0, "Column blocksize should be fully divided by the number of rows"
                self.blocknum = self.columns // blocksize
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
            self.sparsity_probabilities = nn.Parameter(torch.zeros((self.blocknum, self.sparsity_candidates.shape[0]), device=self.device))
        else:
            self.sparsity_probabilities = nn.Parameter(torch.zeros_like(self.sparsity_candidates, device=self.device))
        self.update_sparsity()

        map_dim_size = self.columns if self.wise_dim == 'row' else self.rows if self.wise_dim == 'column' else -1
        self.prob_map_matrix = torch.zeros((len(self.sparsity_candidates), map_dim_size), device=self.device)
        for i in range(len(self.sparsity_candidates)):
            self.prob_map_matrix[i, :int(map_dim_size * self.sparsity_candidates[i].item())] = 1

        self.use_lora = lora_rank != -1
        if self.use_lora:
            assert type(lora_rank) is int and 0 < lora_rank < min(self.rows, self.columns), f"Invalid Lora rank: {lora_rank}"
            self.lora_A = nn.Parameter(torch.zeros((lora_rank, self.columns), device=self.device))
            self.lora_B = nn.Parameter(torch.zeros((self.rows, lora_rank), device=self.device))
            self.lora_scaling = lora_alpha / lora_rank

    def finish_learn_sparsity(self):
        if self.learn_sparsity:
            if self.use_lora:
                lora_weight = (self.lora_B.data @ self.lora_A.data).detach() * self.lora_scaling
                self.weight.data += lora_weight.to(self.weight.dtype)
                self.lora_A = None
                self.lora_B = None
                self.lora_scaling = None
            self.update_sparsity()
            self.prune_mask = self.get_prune_mask().detach()
            self.weight.data *= self.prune_mask
        self.learn_sparsity = False

        self.W_metric_sort_indices = None
        self.sparsities = None
        self.prob_map_matrix = None
        self.sparsity_candidates = None
        self.sparsity_probabilities = None
        self.sparsity_probabilities_softmax = None
        torch.cuda.empty_cache()

    def update_sparsity(self):
        if self.sigmoid_smooth:
            self.sparsity_probabilities_softmax = self.sparsity_probabilities.sigmoid().softmax(dim=-1)
        else:
            self.sparsity_probabilities_softmax = self.sparsity_probabilities.softmax(dim=-1)
        if self.block_wise:
            self.sparsities = self.sparsity_probabilities_softmax @ self.sparsity_candidates
            self.sparsity = self.sparsities.mean()
        else:
            self.sparsity = torch.matmul(self.sparsity_candidates, self.sparsity_probabilities_softmax)
        return self.sparsity

    def get_weight_mask(self):
        W_mask = torch.ones((self.rows, self.columns), device=self.device)
        if self.prune_n != 0:
            # structured n:m sparsity
            for ii in range(self.columns):
                if ii % self.prune_m == 0:
                    tmp = self.W_metric[:, ii:(ii + self.prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], 0)
        elif self.block_wise:
            # block wise unstructured pruning
            if self.wise_dim == 'row':
                row_block_prune_num = (self.sparsities * self.columns).to(dtype=torch.long)
                row_prune_num = row_block_prune_num.reshape((-1, 1)).repeat(1, self.blocksize).reshape(-1)
                W_mask = mask_gen_cuda.mask_gen_forward(W_mask, self.W_metric_sort_indices, row_prune_num)[0]
            elif self.wise_dim == 'column':
                column_block_prune_num = (self.sparsities * self.rows).to(dtype=torch.long)
                column_prune_num = column_block_prune_num.reshape((-1, 1)).repeat(1, self.blocksize).reshape(-1)
                W_mask = mask_gen_cuda.mask_gen_forward(W_mask.t().contiguous(), self.W_metric_sort_indices.t().contiguous(), column_prune_num)[0]
                W_mask = W_mask.t().contiguous()
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
        else:
            # unstructured pruning
            if self.wise_dim == 'row':
                indices = self.W_metric_sort_indices[:, :int(self.columns * self.sparsity)]
                W_mask.scatter_(1, indices, 0)
            elif self.wise_dim == 'column':
                indices = self.W_metric_sort_indices[:int(self.rows * self.sparsity), :]
                W_mask.scatter_(0, indices, 0)
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
        return W_mask

    def get_prob_mask(self):
        P_mask = torch.zeros((self.rows, self.columns), device=self.device)
        probabilities = 1 - (self.sparsity_probabilities_softmax @ self.prob_map_matrix)
        if not self.block_wise:
            if self.wise_dim == 'row':
                probabilities = probabilities.repeat(self.rows, 1)
            elif self.wise_dim == 'column':
                probabilities = probabilities.reshape((-1, 1)).repeat(1, self.columns)
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
        else:
            if self.wise_dim == 'row':
                probabilities = probabilities.reshape((self.blocknum, 1, self.columns))
                probabilities = probabilities.repeat(1, self.blocksize, 1)
            elif self.wise_dim == 'column':
                probabilities = probabilities.reshape((self.rows, self.blocknum, 1))
                probabilities = probabilities.repeat(1, 1, self.blocksize)
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
            probabilities = probabilities.reshape((self.rows, self.columns))
        probabilities = probabilities.to(dtype=P_mask.dtype)
        scatter_dim = 1 if self.wise_dim == 'row' else 0 if self.wise_dim == 'column' else -1
        P_mask.scatter_(scatter_dim, self.W_metric_sort_indices, probabilities)
        return P_mask

    def get_prune_mask(self):
        W_mask = self.get_weight_mask()
        P_mask = self.get_prob_mask()
        prune_mask = W_mask.detach() - P_mask.detach() + P_mask
        prune_mask = prune_mask.to(dtype=self.weight.dtype)
        return prune_mask

    def forward(self, input: torch.Tensor):
        weight = self.weight.detach()
        if self.learn_sparsity:
            self.update_sparsity()
            prune_mask = self.get_prune_mask()
            if self.use_lora:
                lora_weight = (self.lora_B @ self.lora_A) * self.lora_scaling
                weight += lora_weight.to(dtype=self.weight.dtype)
            if self.use_error:
                weight = torch.mul(self.quantization_error, prune_mask)
            else:
                weight = torch.mul(weight, prune_mask)
        out = self.linear_func(input, weight, self.bias)
        return out



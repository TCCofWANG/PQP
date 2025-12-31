# PQP
![intuition](figures/overview.png)

This repository contains code to reproduce the key results of the paper PQP: Joint Post-training Quantization and Pruning for Efficient LLM
Compression.

## Dependencies

* `torch`: tested on v2.0.1+cu118
* `transformers`: tested on v4.46.1
* `accelerate`: tested on v0.34.2
* `datasets`: tested on v4.4.2
* `timm`: tested on v0.9.5

**lm-evaluation-harness**
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

**Customized Cuda Operator**
```
cd models/ops
python setup.py install
```

## Usage

Here is the command to run baseline experiments followed by perplexity evaluations on WikiText2, C4 and zero-shot tasks.

```
python3 main.py
```

## Others
In the experiment section of our paper, we present the results of row-wise sparsity, which customize sparsity for each row of target layer's weight within in the block. Additionally, we provide an extension presenting the outcomes of layer-wise sparsity, where each row of the target layer is assigned uniform sparsity. Below, we present the perplexity results for the Wikitext2 dataset.

| Method            | 1-7B | 1-13B | 1-30B | 1-65B | 2-7B | 2-13B | 2-70B | 3.1-8B | 
|------------------:|:-----|:------|:------|:------|:-----|:------|:------|:------ |
|SparseGPT          | 9.21 | 6.50  | 5.55  | 4.84  | 7.72 | 6.29  | 4.48  | 92.41  |
|Wanda              | 7.98 | 6.58  | 5.60  | 4.92  | 7.53 | 6.40  | 4.58  | 11.67  |
|RIA                | 8.21 | 6.60  | 5.55  | 4.91  | 7.67 | 6.35  | 4.61  | 11.41  |
|BESA               | 7.38 | 6.40  | 5.34  | 4.67  | 7.55 | 6.19  | 4.47  | 11.01  |
|BaWA               | 8.10 | 6.65  | 5.55  | 4.91  | 7.58 | 6.33  | 4.59  | 11.02  |
|JSQ                | 7.84 | 6.52  | 5.76  | 5.10  | 7.56 | 6.41  | 4.61  | 11.12  |
|PQP (layer-wise)   |7.67(↑4%) |6.31(↓1%) |5.30(↓1%) |4.61(↓1%) |7.17(↓5%) |6.03(↓3%) |4.29(↓4%) |11.47(↑4%)|
|PQP (row-wise)     |7.08(↓4%) |6.07(↓5%) |5.10(↓4%)| 4.45(↓5%) |6.89(↓8%)| 5.84(↓6%) |4.23(↓5%)| 9.67(↓12%)|

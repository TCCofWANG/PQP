import random
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_from_disk("/public/lzy/dataset/wikitext-2-raw-v1/train/")
    testdata = load_from_disk("/public/lzy/dataset/wikitext-2-raw-v1/test/")

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_from_disk("/public/lzy/dataset/ptb/train/")
    testdata = load_from_disk("/public/lzy/dataset/ptb/test/")
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_from_disk("/public/lzy/dataset/c4/train/")
    valdata = load_from_disk("/public/lzy/dataset/c4/validation/")
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, tokenizer=None
):
    print(f"Loaded data from {name};")
    if name == 'wikitext2':
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name == 'ptb':
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if name == 'c4':
        return get_c4(nsamples, seed, seqlen, tokenizer)
    else:
        raise NotImplementedError(f"Invalid dataset name: {name}")

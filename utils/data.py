import random
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # traindata = load_from_disk("/home/user0/public2/lzy/data/wikitext-2-raw-v1/train/")
    # testdata = load_from_disk("/home/user0/public2/lzy/data/wikitext-2-raw-v1/test/")
    traindata = load_from_disk("/public/lzy/dataset/wikitext-2-raw-v1/train/")
    testdata = load_from_disk("/public/lzy/dataset/wikitext-2-raw-v1/test/")
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

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
    # traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    # testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    # traindata = load_from_disk("/home/user0/public2/lzy/data/ptb/train/")
    # testdata = load_from_disk("/home/user0/public2/lzy/data/ptb/test/")
    traindata = load_from_disk("/public/lzy/dataset/ptb/train/")
    testdata = load_from_disk("/public/lzy/dataset/ptb/test/")
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

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
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    # traindata = load_from_disk("/home/user0/public2/lzy/data/c4/train/")
    # valdata = load_from_disk("/home/user0/public2/lzy/data/c4/validation/")
    traindata = load_from_disk("/public/lzy/dataset/c4/train/")
    valdata = load_from_disk("/public/lzy/dataset/c4/validation/")
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

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
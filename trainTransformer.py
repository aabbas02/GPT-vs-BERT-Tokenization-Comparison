import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import  AutoTokenizer,GPT2Tokenizer
from utilsTrainTransformer import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = "bert"
if tokenizer == "gpt":
    #tknzr_de = AutoTokenizer.from_pretrained("Xenova/gpt-4")
    tknzr_de = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
else:
    tokenizer = "bert"
    tknzr_de = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

tknzr_en = tknzr_de

# update URL links to the datasets because the links given in the Harvard NLP code do not work
from torchtext.datasets import multi30k
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"
#---------------------------
multi30k.MD5["train"] = "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e"
multi30k.MD5["valid"] = "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c"
multi30k.MD5["test"] = "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36"

warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

def is_interactive_notebook():
    return __name__ == "__main__"

def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


def build_vocabulary(tknzr_de, tknzr_en):
    def tokenize_de(text):
        return tokenize(text, tknzr_de)

    def tokenize_en(text):
        return tokenize(text, tknzr_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=1,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=1,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab():
    if not exists("vocab_%s.pt" %tokenizer):
        vocab_src, vocab_tgt = build_vocabulary(tknzr_de, tknzr_en)
        torch.save((vocab_src, vocab_tgt), "vocab_%s.pt" %tokenizer)
    else:
        vocab_src, vocab_tgt = torch.load("vocab_%s.pt" %tokenizer)
    print("Finished.Vocabulary sizes:")
    print(f"German Vocabulary Length = {len(vocab_src)}")
    print(f"English Vocabulary Length = {len(vocab_tgt)}")
    return vocab_src, vocab_tgt
    
vocab_src, vocab_tgt = load_vocab()



def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 15,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_%s" %tokenizer
    }
    model_path = "%s_final.pt" %config["file_prefix"]
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, tknzr_de, tknzr_en, config,device)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(model_path,map_location=device))
    return model


if is_interactive_notebook():
    model = load_trained_model()


if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight

def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=15):
    global vocab_src, vocab_tgt, tknzr_de, tknzr_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
    	device,
        vocab_src,
        vocab_tgt,
        tknzr_de, 
        tknzr_en, 
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.to(device)

    model_path = "multi30k_model_%s_final.pt" %tokenizer
    model.load_state_dict(
        torch.load(model_path, map_location = device )
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


execute_example(run_model_example)



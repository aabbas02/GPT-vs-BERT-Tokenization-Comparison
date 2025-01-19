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

"""
# Set to False to skip notebook execution (e.g. for debugging)
"""
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

# Some convenience helper functions used throughout the notebook
def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)

def collate_batch(
    batch,
    vocabBertDe,
    vocabBertEng,
    vocabGptDe,
    vocabGptEng,
    device,
    max_padding=72,
    pad_id=2,
    tknzrBert=None,
    tknzrGpt=None
):
    #------------------------------
    # BERT TOKENIZATION AND PADDING
    #------------------------------
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch: # this for loop, which iterates over the number of data points, can be removed 
        ids = tknzrBert(_src)['input_ids']
        srcTokens =  [tknzrBert.decode(ids[i]) for i in range(len(ids))]
        ids = tknzrBert(_tgt)['input_ids']
        tgtTokens =  [tknzrBert.decode(ids[i]) for i in range(len(ids))]
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    vocabBertDe(srcTokens), #vocabBertDe(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    vocabBertEng(tgtTokens),#vocabBertEng(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )
    srcBert = torch.stack(src_list)
    tgtBert = torch.stack(tgt_list)
    #-----------------------------
    # GPT TOKENIZATION AND PADDING
    #-----------------------------
    src_list, tgt_list = [], []
    #------------
    for (_src,_tgt) in batch: # this for loop, which iterates over the number of data points, can be removed
        ids = tknzrGpt(_src)['input_ids']
        srcTokens =  [tknzrGpt.decode(ids[i]) for i in range(len(ids))]
        ids = tknzrGpt(_tgt)['input_ids']
        tgtTokens =  [tknzrGpt.decode(ids[i]) for i in range(len(ids))]
        processed_src = torch.cat(
            [bs_id,
                #torch.unsqueeze(torch.tensor(bs_id,device=device),dim=0),
                torch.tensor(
                    vocabGptDe(srcTokens),
                    dtype=torch.int64,
                    device=device,
                ),
            eos_id
                #torch.unsqueeze(torch.tensor(eos_id,device=device),dim=0),
            ],
            0,
        )
        processed_tgt = torch.cat(
            [bs_id,
                #torch.unsqueeze(torch.tensor(bs_id,device=device),dim=0),
                torch.tensor(
                    vocabGptEng(tgtTokens),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id
                #torch.unsqueeze(torch.tensor(eos_id,device=device),dim=0),
                                        ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )
    srcGpt = torch.stack(src_list)
    tgtGpt = torch.stack(tgt_list)
    return (srcBert, tgtBert, srcGpt, tgtGpt)

def create_dataloaders(
    device,
    vocabBertDe,
    vocabBertEng,
    vocabGptDe,
    vocabGptEng,
    batch_size = 12000,
    max_padding = 72,
    is_distributed = True,
    tknzrBert = None,
    tknzrGpt = None
):

    def collate_fn(batch):
        return collate_batch(
            batch,
            vocabBertDe,
            vocabBertEng,
            vocabGptDe,
            vocabGptEng,
            device,
            max_padding=max_padding,
            pad_id=vocabBertDe.get_stoi()["<blank>"], #probably 2
            tknzrBert=tknzrBert,
            tknzrGpt=tknzrGpt
            )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    #train_iter_map = to_map_style_dataset(
    #    train_iter
    #)  # DistributedSampler needs a dataset len()
    #train_sampler = (
    #    DistributedSampler(train_iter_map) if is_distributed else None
    #)
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )
    """
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    """
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return valid_dataloader #train_dataloader, valid_dataloader



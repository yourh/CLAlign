#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm.auto import tqdm
from typing import Sequence

from clalign.alignment import ProteinSeq

__all__ = ['get_embs']


class PLMDataset(Dataset):
    """

    """

    def __init__(self, seqs, tokenizer, max_length=512):
        super().__init__()
        chunk_size = max_length - (tokenizer.cls_token is not None) - (tokenizer.eos_token is not None)
        self.inputs = []
        for seq in tqdm(seqs, desc='Tokenize', leave=False, dynamic_ncols=True, delay=10):
            for i in range(0, len(seq), chunk_size):
                inputs_ = tokenizer(' '.join(seq[i: i + chunk_size]), return_tensors='pt')
                self.inputs.append({k: v[0] for k, v in inputs_.data.items()})

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item]


@torch.no_grad()
def get_embs(seqs: Sequence[ProteinSeq | str | list[str]], tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
             max_length: int, enable_amp: bool = False, num_models: int = 1, pack_emb: bool = False):
    dataloader, embs = DataLoader(PLMDataset(seqs, tokenizer, max_length), batch_size=1), []
    model.eval()
    i = 0
    for batch_inputs in tqdm(dataloader, desc='Getting Embeddings', leave=False, dynamic_ncols=True, delay=10):
        with torch.autocast(model.device.type, enabled=enable_amp):
            output_ = model(**{k: v.to(model.device) for k, v in batch_inputs.items()}).last_hidden_state[0]
            output_ = F.normalize(output_, dim=-1)
            if tokenizer.cls_token is not None:
                output_ = output_[1:]
            if tokenizer.eos_token is not None:
                output_ = output_[:-1]
                embs.append(output_.cpu())
    embs = np.vstack(embs)
    assert embs.shape[0] == sum([len(x) for x in seqs])
    if pack_emb:
        return embs
    idx = np.cumsum([0] + [len(x) for x in seqs])
    return [embs[s:t] for s, t in zip(idx[:-1], idx[1:])]

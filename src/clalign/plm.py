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
from esm.inverse_folding.util import load_coords, CoordBatchConverter
from tqdm.auto import tqdm
from typing import Sequence

from clalign.alignment import ProteinSeq

__all__ = ['PLM']


class PLMDataset(Dataset):
    """

    """

    def __init__(self, seqs, tokenizer, model, max_length=512):
        super().__init__()
        if tokenizer is not None:
            self.max_length = max_length - (tokenizer.cls_token is not None) - (tokenizer.eos_token is not None)
        else:
            self.max_length, self.batch_converter = max_length, CoordBatchConverter(model.alphabet)
        chunk_size = self.max_length // 2
        half_size = chunk_size // 2
        self.tokenizer, self.inputs = tokenizer, []
        for seq in tqdm(seqs, total=len(seqs), desc='Tokenize', leave=False, dynamic_ncols=True, delay=10):
            if len(seq) <= self.max_length:
                self.inputs.append((seq, (0, len(seq))))
            else:
                self.inputs.append((seq[:chunk_size * 2], (0, chunk_size)))
                for s in range(chunk_size, len(seq), chunk_size):
                    if s + chunk_size + half_size <= len(seq):
                        assert s - half_size >= 0
                        self.inputs.append((seq[s - half_size: s + chunk_size + half_size],
                                            (half_size, half_size + chunk_size)))
                    else:
                        x = s + chunk_size + half_size - len(seq)
                        assert s - half_size - x >= 0
                        s_ = seq[s - half_size - x:]
                        self.inputs.append((s_, (half_size + x, min(half_size + x + chunk_size, len(s_)))))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item]

    def collate_fn(self, batch):
        if self.tokenizer is not None:
            inputs = self.tokenizer([' '.join(x[0]) if 't5' in self.tokenizer.name_or_path.lower() else x[0]
                                     for x in batch], padding=True, return_tensors='pt')
        else:
            inputs = self.batch_converter.from_lists([x[0] for x in batch])
        return inputs, (torch.as_tensor([x[1][0] for x in batch]), torch.as_tensor([x[1][1] for x in batch]))


class PLM(object):
    """

    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, max_length: int,
                 enable_amp : bool = True, batch_size : int = 1):
        self.tokenizer, self.model, self.max_length = tokenizer, model, max_length
        self.enable_amp, self.batch_size = enable_amp, batch_size

    @torch.no_grad()
    def get_embs(self, seqs: Sequence[ProteinSeq | np.ndarray | str | list[str]], batch_size=1,
                 pack_emb: bool = False):
        dataloader = DataLoader(dataset:=PLMDataset(seqs, self.tokenizer, self.model, self.max_length),
                                batch_size=batch_size, collate_fn=dataset.collate_fn)
        embs = []
        self.model.eval()
        for batch_inputs, (s, t) in tqdm(dataloader, desc='Getting Embeddings', leave=False,
                                         dynamic_ncols=True, delay=10):
            with torch.autocast(self.model.device.type, enabled=self.enable_amp):
                output_ = F.normalize(self.model(**{k: v.to(self.model.device)
                                                    for k, v in batch_inputs.items()}).last_hidden_state,
                                      dim=-1)
                if self.tokenizer is None or self.tokenizer.cls_token is not None:
                    output_ = output_[:, 1:]
                if self.tokenizer is None or self.tokenizer.eos_token is not None:
                    output_ = output_[:, :-1]
                for o_, s_, t_ in zip(output_, s, t):
                    embs.append(o_[s_:t_].cpu())
        embs = np.vstack(embs) if embs else np.empty((0, 0))
        assert embs.shape[0] == sum([len(x) for x in seqs])
        if pack_emb:
            return embs
        idx = np.cumsum([0] + [len(x) for x in seqs])
        return [embs[s:t] for s, t in zip(idx[:-1], idx[1:])]

    @staticmethod
    def unpack_emb(seqs, packed_embs):
        emb_st = np.cumsum(np.asarray([0] + [len(x) for x in seqs], dtype=np.int32))
        return [packed_embs[emb_st[i]: emb_st[i + 1]] for i in range(len(seqs))]

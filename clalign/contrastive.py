#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer, PreTrainedModel, T5Tokenizer, EsmModel, T5EncoderModel
from tqdm.auto import tqdm
from logzero import logger

__all__ = ['is_master', 'train']


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0


class CLDataset(Dataset):
    """
    
    """

    def __init__(self, data_file, tokenizer: PreTrainedTokenizer, max_length=512):
        super().__init__()
        max_length -= (tokenizer.cls_token is not None) + (tokenizer.eos_token is not None)
        self.tokenizer, self.data = tokenizer, []
        with open(data_file) as fp:
            for line in tqdm(fp, desc='Reading Data', leave=False, dynamic_ncols=True, delay=10, disable=not is_master()):
                *_, seq1, seq2, aln = line.strip().split('\t')
                if len(seq1) > max_length or len(seq2) > max_length:
                    continue
                aln = torch.as_tensor(self.get_matched_res(aln)).T
                self.data.append(((' '.join(seq1), aln[0]), (' '.join(seq2), aln[1])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        inputs = self.tokenizer([x[0][0] for x in batch] + [x[1][0] for x in batch], padding=True, return_tensors='pt')
        aln = [y + i * inputs['input_ids'].shape[1] + (self.tokenizer.cls_token is not None)
               for i, y in enumerate([x[0][1] for x in batch] + [x[1][1] for x in batch])]
        return inputs, torch.cat(aln)

    @staticmethod
    def get_matched_res(aln_text):
        i, j, pos = 0, 0, []
        for x in aln_text:
            match x:
                case ':': pos, i, j = pos + [(i, j)], i + 1, j + 1
                case '1': i += 1
                case '2': j += 1
                case '.': i, j = i + 1, j + 1
                case _: raise ValueError
        return pos


class ContrastiveLoss(nn.Module):
    """

    """

    def __init__(self, n_views=2, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ce_loss, self.n_views, self.temperature = nn.CrossEntropyLoss(), n_views, temperature

    class GatherLayer(torch.autograd.Function):
        """
        """

        @staticmethod
        def forward(ctx, inputs):
            ws = dist.get_world_size()
            dist.all_gather_into_tensor(sizes := torch.zeros(ws, dtype=torch.int, device=inputs.device),
                                        torch.as_tensor(inputs.shape[0], dtype=torch.int, device=inputs.device))
            dist.all_gather(outputs := [torch.zeros(sizes[i].item(), *inputs.shape[1:],
                                                    device=inputs.device, dtype=inputs.dtype) for i in range(ws)],
                            inputs)
            return tuple(outputs)

        @staticmethod
        def backward(ctx, *grads):
            return grads[dist.get_rank()] * dist.get_world_size()

    def forward(self, embeddings):
        embeddings = F.normalize(embeddings, dim=-1)
        labels = torch.arange(n_ := embeddings.shape[0], device=embeddings.device) % (n_ // self.n_views)
        if dist.is_initialized():
            embeddings = torch.vstack(outputs := self.GatherLayer.apply(embeddings))
            labels += np.cumsum([0] + [x.shape[0] for x in outputs])[dist.get_rank()]
            dist.all_gather(labels_list := [torch.zeros(outputs[i].shape[0], dtype=labels.dtype,
                                                        device=labels.device) for i in range(len(outputs))], labels)
            labels = torch.hstack(labels_list)
        labels = labels[:, None] == labels[None]

        similarity_matrix = embeddings @ embeddings.T
        mask = torch.eye(*labels.shape, dtype=torch.bool, device=embeddings.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels].view(-1, 1)
        negatives = similarity_matrix[~labels].view(similarity_matrix.shape[0], -1)
        negatives = negatives[:, None].expand(-1, self.n_views - 1, -1).flatten(0, 1)
        logits = torch.hstack([positives, negatives]) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=embeddings.device)

        return self.ce_loss(logits, labels)

def train(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, model_path: Path | str, train_data: Path | str,
          max_length: int, temperature: float, num_epochs: int, batch_size: int,
          learning_rate: float, weight_decay: float, enable_amp: bool):
    logger.info(f'batch_size={batch_size * dist.get_world_size() if dist.is_initialized() else 1}')
    logger.info(f'num_epochs={num_epochs}')
    logger.info(f'learning_rate={learning_rate}')
    logger.info(f'weight_decay={weight_decay}')
    train_loader = DataLoader(d_ := CLDataset(train_data, tokenizer, max_length), batch_size, collate_fn=d_.collate_fn,
                              shuffle=True if not dist.is_initialized() else None,
                              sampler=None if not dist.is_initialized() else DistributedSampler(d_))
    dp_network = nn.parallel.DistributedDataParallel(model) if dist.is_initialized() else model
    opt = torch.optim.AdamW(dp_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = ContrastiveLoss(temperature=temperature)
    scaler = torch.GradScaler(enabled=enable_amp)
    dp_network.train()
    for epoch_idx in range(num_epochs):
        for inputs, aln in tqdm(train_loader, desc=f'Epoch {epoch_idx}', leave=False,
                                dynamic_ncols=True, delay=10, disable=not is_master()):
            with torch.autocast(dp_network.device.type, enabled=enable_amp):
                inputs = {k: v.to(dp_network.device) for k, v in inputs.items()}
                embs = dp_network(**inputs).last_hidden_state.flatten(0, 1)[aln.to(dp_network.device)]
                loss = loss_fn(embs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            if is_master():
                tqdm.write(f'{loss.item():.5f}')
    if dist.is_initialized():
        dist.barrier()
    model.save_pretrained(model_path, is_main_process=is_master())

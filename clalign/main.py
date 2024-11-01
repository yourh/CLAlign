#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

import os
import math
import csv
import importlib.resources
import click
import Bio.SeqIO
import numpy as np
import torch
import torch.distributed as dist
import transformers
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig
from logzero import logger

from clalign.plm import *
from clalign.alignment import *
from clalign.metrics import *
from clalign.contrastive import *


def get_resource_path(package='clalign', resource='CLAlignT5'):
    with importlib.resources.path(package, resource) as resource_path:
        return resource_path


@click.group()
@click.option('--tokenizer-cls-name', default='T5Tokenizer', show_default=True)
@click.option('--model-cls-name', default='T5EncoderModel', show_default=True)
@click.option('--model-name', default='Rostlab/prot_t5_xl_uniref50', show_default=True)
@click.option('--model-path', type=Path, default=get_resource_path(), show_default=True)
@click.option('--lora', is_flag=True)
@click.option('--max-length', default=2048, show_default=True)
@click.option('-d', '--device', default='cuda', show_default=True)
@click.option('-a', '--amp', 'enable_amp', is_flag=True)
@click.pass_context
def main(ctx, tokenizer_cls_name, model_cls_name, model_name, model_path, lora, max_length, device, enable_amp):
    ctx.ensure_object(dict)

    tokenizer = getattr(transformers, tokenizer_cls_name,
                        AutoTokenizer).from_pretrained(model_name, use_fast=False, legacy=False)
    model = getattr(transformers, model_cls_name, AutoModel).from_pretrained(model_name)
    if ctx.invoked_subcommand != 'cl':
        if lora and ctx.invoked_subcommand != 'cl' and model_path.exists():
            model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
            logger.info(f'Load existing model from {model_path}')
        model = model.to(device)

    ctx.obj['args'] = {
        'device': device,
        'tokenizer': tokenizer,
        'model': model,
        'model_path': model_path,
        'max_length': max_length,
        'enable_amp': enable_amp
    }

@main.command()
@click.argument('db_fasta', type=Path)
@click.argument('db_path', type=Path, default=None, required=False)
@click.pass_context
def create_db(ctx, db_fasta: Path, db_path: Path):
    args = ctx.obj['args']
    if db_path is None:
        db_path = db_fasta.with_suffix('.claligndb')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    embs = get_embs([ProteinSeq(str(x.seq)) for x in Bio.SeqIO.parse(db_fasta, 'fasta')],
                    args['tokenizer'], args['model'], args['max_length'], args['enable_amp'], pack_emb=True)
    np.save(db_path, embs)


@main.group()
@click.option('-g', '--gap-penalty', type=click.FLOAT, default=0.0, show_default=True)
@click.pass_context
def align(ctx, gap_penalty):
    args = ctx.obj['args']
    args['gap_penalty'] = gap_penalty


@align.command()
@click.argument('query_fasta', type=Path)
@click.argument('db_fasta', type=Path)
@click.argument('output_path', type=Path)
@click.option('--qe', 'query_embs', type=Path, default=None)
@click.option('--de', 'db_embs', type=Path, default=None)
@click.option('-k', '--keep', type=click.FLOAT, default=-1, show_default=True)
@click.option('--only-score', is_flag=True)
@click.pass_context
def query(ctx, query_fasta: Path, db_fasta: Path, output_path: Path, query_embs: Path, db_embs: Path, keep, only_score):
    args = ctx.obj['args']

    def get_embs_(seqs, embs):
        if embs is not None:
            return np.load(embs)
        return get_embs(seqs, args['tokenizer'], args['model'], args['max_length'], args['enable_amp'], pack_emb=True)

    query_seqs = [ProteinSeq(str(x.seq), x.id) for x in Bio.SeqIO.parse(query_fasta, 'fasta')]
    db_seqs = [ProteinSeq(str(x.seq), x.id) for x in Bio.SeqIO.parse(db_fasta, 'fasta')]
    db_embs = get_embs_(db_seqs, db_embs)
    query_embs = db_embs if query_fasta.samefile(db_fasta) else get_embs_(query_seqs, query_embs)
    if args['model'].device.type == 'cuda':
        del args['model']
        torch.cuda.empty_cache()
    keep = len(db_seqs) if keep < 0 else min(math.ceil(len(db_seqs) * keep if keep < 1 else keep), len(db_seqs))
    aln_res = pairwise_align(query_seqs, query_embs, db_seqs, db_embs, keep, args['gap_penalty'], only_score)
    if output_path.suffix != '.csv':
        output_path = output_path.with_name(f'{output_path.name}.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['query_id', 'db_id', 'query_aln', 'db_aln', 'query_start', 'db_start', 'score'])
        for res_ in aln_res:
            for r_ in sorted(res_, reverse=True):
                writer.writerow([r_.seq1.pid, r_.seq2.pid, r_.aln1, r_.aln2, r_.start1, r_.start2, r_.score])


@align.command()
@click.argument('inputs', type=Path)
@click.pass_context
def pair(ctx, inputs):
    args = ctx.obj['args']
    with open(inputs) as fp:
        seq1, seq2 = fp.readline().strip().upper(), fp.readline().strip().upper()
    seq1, seq2 = ProteinSeq(seq1), ProteinSeq(seq2)
    embs = get_embs([seq1, seq2], args['tokenizer'], args['model'], args['max_length'], args['enable_amp'])
    aln_res = fast_align(seq1, seq2, embs[0] @ embs[1].T, args['gap_penalty'])
    print(f'Alignment Score: {aln_res.score:.6f}')
    print()
    draw_alignment(aln_res)


@main.command()
@click.option('--train-data', type=Path, default='data/train10k.txt', show_default=True)
@click.option('-r', '--rank', type=click.INT, default=8, show_default=True)
@click.option('--target-modules', type=click.STRING, default='q,k,v,o,wi,wo', show_default=True)
@click.option('--modules-to-save', type=click.STRING, default='layer_norm', show_default=True)
@click.option('-t', '--temperature', type=click.FLOAT, default=0.1, show_default=True)
@click.option('-e', '--num-epochs', type=click.INT, default=3, show_default=True)
@click.option('-b', '--batch-size', default=1, show_default=True)
@click.option('--lr', 'learning_rate', type=click.FLOAT, default=2e-5, show_default=True)
@click.option('-w', '--weight-decay', type=click.FLOAT, default=1e-2, show_default=True)
@click.option('--dist', 'enable_dist', is_flag=True)
@click.pass_context
def cl(ctx, train_data, rank, target_modules, modules_to_save, temperature, num_epochs, batch_size,
       learning_rate, weight_decay, enable_dist):

    if enable_dist:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        logger.info(f'Using DDP in {os.uname()[1]} with rank {dist.get_rank()}, '
                    f'PID: {os.getpid()}, PPID: {os.getppid()}')
        if dist.get_rank() > 0:
            logger.setLevel(100)
        dist.barrier()

    args = ctx.obj['args']
    model = get_peft_model(args['model'], LoraConfig(r=rank, target_modules=target_modules.split(','),
                                                     modules_to_save=modules_to_save.split(','))).to(args['device'])
    if is_master():
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n)
        model.print_trainable_parameters()
    train(args['tokenizer'], model, args['model_path'], train_data, args['max_length'],
          temperature, num_epochs, batch_size, learning_rate, weight_decay, args['enable_amp'])


if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

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
from esm.inverse_folding.util import load_coords
from pathlib import Path
from peft import get_peft_model, PeftModel, LoraConfig
from logzero import logger, loglevel

from clalign.plm import *
from clalign.alignment import *
from clalign.metrics import *
from clalign.contrastive import *
from clalign.utils import *

def get_resource_path(package='clalign', resource='CLAlign-ProtT5'):
    with importlib.resources.path(package, resource) as resource_path:
        return resource_path


def get_inputs(inputs_path, pdb_dir: Path = None, suffix='.pdb'):
    if pdb_dir and pdb_dir.is_dir() and pdb_dir.exists():
        inputs, seqs = [], []
        with open(inputs_path) as fp:
            for i, line in enumerate(fp):
                name = line.strip()
                coords, seq = load_coords(str(pdb_dir / f'{name}{suffix}'))
                inputs.append(coords)
                seqs.append(ProteinSeq(seq, name, '', i))
    else:
        seqs = [ProteinSeq(str(x.seq), x.id, x.description, i)
                for i, x in enumerate(Bio.SeqIO.parse(inputs_path, 'fasta'))]
        inputs = [x.seq for x in seqs]
    return inputs, seqs


@click.group()
@click.option('--tokenizer-cls-name', default='T5Tokenizer', show_default=True)
@click.option('--tokenizer-name', default=None, show_default=True)
@click.option('--base-model-cls-name', default='T5EncoderModel', show_default=True)
@click.option('--base-model-name', default='Rostlab/prot_t5_xl_uniref50', show_default=True)
@click.option('--model-path', type=Path, default=get_resource_path(), show_default=True)
@click.option('--pdb-dir', type=Path, default=None)
@click.option('--suffix', default='.pdb', show_default=True)
@click.option('--lora/--no-lora', 'use_lora', default=True, show_default=True)
@click.option('--max-length', default=512, show_default=True)
@click.option('-b', '--batch-size', default=1, show_default=True)
@click.option('-d', '--device', default='cuda', show_default=True)
@click.option('-a', '--amp', 'enable_amp', is_flag=True)
@click.option('-q', '--quiet', is_flag=True)
@click.option('--use-precomputed', is_flag=True)
@click.pass_context
def main(ctx, tokenizer_cls_name, tokenizer_name, base_model_cls_name, base_model_name, model_path, pdb_dir, suffix,
         use_lora, max_length, batch_size, device, enable_amp, quiet, use_precomputed):
    if quiet:
        loglevel('ERROR')

    ctx.ensure_object(dict)
    if not use_precomputed:
        tokenizer, base_model = get_base_model(tokenizer_cls_name, tokenizer_name, base_model_cls_name, base_model_name)
        if ctx.invoked_subcommand != 'cl':
            if not model_path.exists():
                model_path = get_resource_path('clalign', model_path)
            if use_lora:
                logger.info(f'Load existing model from {model_path}')
                model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
            else:
                model = base_model
                if model_path.exists():
                    logger.info(f'Load existing model from {model_path}')
                    model.load_state_dict(torch.load(model_path/'model.pt', map_location='cpu'))
            model = model.to(device)
            plm = PLM(tokenizer, model, max_length, enable_amp)
        else:
            model = plm = None
    else:
        tokenizer = base_model = model = plm = None

    ctx.obj['args'] = {
        'device': device,
        'tokenizer': tokenizer,
        'base_model': base_model,
        'model': model,
        'model_path': model_path,
        'max_length': max_length,
        'batch_size': batch_size,
        'enable_amp': enable_amp,
        'use_lora': use_lora,
        'plm': plm,
        'pdb_dir': pdb_dir,
        'suffix': suffix
    }


@main.command()
@click.argument('fasta_or_list', type=Path)
@click.argument('emb_path', type=Path, default=None, required=False)
@click.pass_context
def generate_embs(ctx, fasta_or_list: Path, emb_path: Path):
    args = ctx.obj['args']
    if emb_path is None:
        emb_path = fasta_or_list.with_suffix('.clalign')
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    embs = args['plm'].get_embs(get_inputs(fasta_or_list, args['pdb_dir'], args['suffix'])[0], args['batch_size'],
                                pack_emb=True)
    np.save(emb_path, embs)


@main.group()
@click.option('-g', '--gap-penalty', type=click.FLOAT, default=0.0, show_default=True)
@click.pass_context
def align(ctx, gap_penalty):
    args = ctx.obj['args']
    args['gap_penalty'] = gap_penalty


@align.command()
@click.argument('query_fasta_or_list', type=Path)
@click.argument('ref_fasta_or_list', type=Path)
@click.argument('output_path', type=Path)
@click.option('--qe', 'query_embs', type=Path, default=None)
@click.option('--re', 'ref_embs', type=Path, default=None)
@click.option('-k', '--keep', type=click.FLOAT, default=-1, show_default=True)
@click.option('--local', is_flag=True)
@click.option('--only-score', is_flag=True)
@click.option('-t', '--threshold', type=click.FLOAT, default=0.2, show_default=True)
@click.option('--eff-mem', is_flag=True)
@click.pass_context
def query(ctx, query_fasta_or_list: Path, ref_fasta_or_list: Path, output_path: Path, query_embs: Path, ref_embs: Path,
          keep, local, only_score, threshold, eff_mem):
    args = ctx.obj['args']
    plm = args['plm']

    def get_embs(inputs, embs):
        if embs is not None:
            if not embs.exists() and embs.suffix != '.npy':
                embs = embs.with_suffix(f'{embs.suffix}.npy')
            return np.load(embs)
        if plm is None:
            raise click.UsageError('--use-precomputed requires embedding files')
        return plm.get_embs(inputs, args['batch_size'], pack_emb=True)

    ref_inputs, ref_seqs = get_inputs(ref_fasta_or_list, args['pdb_dir'], args['suffix'])
    ref_embs = get_embs(ref_inputs, ref_embs)
    if query_fasta_or_list.samefile(ref_fasta_or_list):
        query_inputs, query_seqs, query_embs = ref_inputs, ref_seqs, ref_embs
    else:
        query_inputs, query_seqs = get_inputs(query_fasta_or_list, args['pdb_dir'], args['suffix'])
        query_embs = get_embs(query_inputs, query_embs)
    if args['model'] is not None and args['model'].device.type == 'cuda':
        del args['model']
        torch.cuda.empty_cache()

    keep = len(ref_seqs) if keep < 0 else min(math.ceil(len(ref_seqs) * keep if keep < 1 else keep), len(ref_seqs))
    logger.info(f'Starting alignment of {len(query_seqs)} query sequences and {len(ref_seqs)} database sequences')
    aln_res = pairwise_align(query_seqs, query_embs, ref_seqs, ref_embs, args['gap_penalty'], local, keep,
                             only_score, eff_mem)

    if output_path.suffix != '.csv':
        output_path = output_path.with_name(f'{output_path.name}.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['query_id', 'ref_id', 'query_aln', 'ref_aln', 'query_start', 'ref_start', 'score'])
        for res_ in aln_res:
            for r_ in sorted(res_, reverse=True):
                if r_.coverage >= threshold:
                    writer.writerow([r_.seq1.pid, r_.seq2.pid, r_.aln1, r_.aln2, r_.start1, r_.start2, r_.score])


@align.command()
@click.argument('inputs', type=Path)
@click.pass_context
def pair(ctx, inputs):
    args = ctx.obj['args']
    plm = args['plm']
    with open(inputs) as fp:
        seq1, seq2 = fp.readline().strip().upper(), fp.readline().strip().upper()
    seq1, seq2 = ProteinSeq(seq1), ProteinSeq(seq2)
    embs = plm.get_embs([seq1, seq2])
    aln_res = align_core(seq1, seq2, embs[0] @ embs[1].T, args['gap_penalty'])
    print(f'Alignment Score: {aln_res.score:.6f}')
    draw_alignment(aln_res)


@main.command()
@click.option('--train-data', type=Path, default='data/train10k.txt', show_default=True)
@click.option('-r', '--rank', type=click.INT, default=8, show_default=True)
@click.option('--target-modules', type=click.STRING, default='q,k,v,o,wi,wo', show_default=True)
@click.option('-t', '--temperature', type=click.FLOAT, default=0.1, show_default=True)
@click.option('-e', '--num-epochs', type=click.INT, default=3, show_default=True)
@click.option('-b', '--batch-size', default=1, show_default=True)
@click.option('--lr', 'learning_rate', type=click.FLOAT, default=2e-5, show_default=True)
@click.option('-w', '--weight-decay', type=click.FLOAT, default=1e-2, show_default=True)
@click.option('--dist', 'enable_dist', is_flag=True)
@click.pass_context
def cl(ctx, train_data, rank, target_modules, temperature, num_epochs, batch_size, learning_rate, weight_decay,
       enable_dist):

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
    if args['use_lora']:
        lora_config = LoraConfig(r=rank, target_modules=target_modules.split(','))
        model = get_peft_model(args['base_model'], lora_config)
    else:
        model = args['base_model']
    model = model.to(args['device'])
    if is_master():
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n)
        if args['use_lora']:
            model.print_trainable_parameters()

    train(args['tokenizer'], model, args['model_path'], args['pdb_dir'], args['suffix'], train_data, args['max_length'],
          temperature, num_epochs, batch_size, learning_rate, weight_decay, args['enable_amp'])

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

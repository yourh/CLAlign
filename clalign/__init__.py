#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

import numba

from clalign.contrastive import train
from clalign.plm import get_embs
from clalign.alignment import ProteinSeq, AlignmentResult, fast_align, pairwise_align
from clalign.metrics import f1score

numba.config.THREADING_LAYER = 'omp'

__all__ = ['ProteinSeq', 'AlignmentResult', 'fast_align', 'pairwise_align', 'f1score', 'train']

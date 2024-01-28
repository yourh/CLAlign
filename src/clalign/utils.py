#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

import transformers
import esm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel

__all__ = ['get_base_model']

def get_base_model(tokenizer_cls_name, tokenizer_name, base_model_cls_name, base_model_name):
    if 'esm_if' in base_model_name.lower() or 'esm-if' in base_model_name.lower():
        base_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        alphabet.prepend_bos = alphabet.append_eos = True
        base_model.encoder.alphabet = alphabet
        return None, base_model.encoder
    tokenizer_cls = getattr(transformers, tokenizer_cls_name, AutoTokenizer)
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_name or base_model_name, use_fast=True, legacy=False)
    model_cls = getattr(transformers, base_model_cls_name, AutoModel)
    try:
        base_model = model_cls.from_pretrained(base_model_name, use_safetensors=True)
    except OSError as exc:
        if 'safetensors' not in str(exc):
            raise exc
        base_model = model_cls.from_pretrained(base_model_name, use_safetensors=False)
    return tokenizer, base_model

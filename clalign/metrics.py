#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

from clalign.alignment import AlignmentResult

__all__ = ['f1score']


def f1score(truth: AlignmentResult, pred: AlignmentResult):
    t_, p_ = set(truth.get_matched_res()), set(pred.get_matched_res())
    pr_, re_ = (x_ := len(t_ & p_)) / (len(p_) + 1e-10), x_ / (len(t_) + 1e-10)
    return pr_, re_, 2 * pr_ * re_ / (pr_ + re_ + 1e-10)

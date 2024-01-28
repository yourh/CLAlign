#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

import numpy as np

from clalign.alignment import AlignmentResult

__all__ = ['f1score', 'hec_acc', 'hec_sov']


def f1score(truth: AlignmentResult, pred: AlignmentResult):
    t_, p_ = set(truth.get_matched_res()), set(pred.get_matched_res())
    pr_, re_ = (x_ := len(t_ & p_)) / (len(p_) + 1e-10), x_ / (len(t_) + 1e-10)
    return pr_, re_, 2 * pr_ * re_ / (pr_ + re_ + 1e-10)


def hec_acc(aln_res, hec1, hec2):
    assert len(aln_res.seq1) == len(hec1)
    assert len(aln_res.seq2) == len(hec2)
    aln1, aln2 = aln_res.aln1, aln_res.aln2
    i = j = s = 0
    for k in range(len(aln1)):
        match (aln1[k], aln2[k]):
            case ('-', _): j += 1
            case (_, '-'): i += 1
            case _: s, i, j = s + (hec1[i] != 'C' and hec1[i] == hec2[j]), i + 1, j + 1
    return s / (0.5 * (len(hec1) + len(hec2)))


def hec_sov(aln_res, hec1, hec2):
    assert len(aln_res.seq1) == len(hec1)
    assert len(aln_res.seq2) == len(hec2)
    seg12 = []
    for hec in (hec1, hec2):
        start, segs = None, []
        for i in range(len(hec) + 1):
            if start is not None and (i == len(hec) or hec[i] != hec[start]):
                if hec[start] == 'H' and i - start >= 3 or hec[start] == 'E' and i - start >= 2:
                    segs.append((start, i))
                start = None
            if start is None and i < len(hec) and hec[i] != 'C':
                start = i
        seg12.append(segs)
    seg1, seg2 = seg12
    if len(seg1) == 0 or len(seg2) == 0:
        return 0.0
    m1 = [-1] * len(hec1)
    for pos in aln_res.get_matched_res():
        m1[pos[0]] = pos[1]
    overlap1, overlap2 = np.zeros(len(seg1)), np.zeros(len(seg2))
    for i in range(len(seg1)):
        for j in range(len(seg2)):
            s1, s2 = seg1[i], seg2[j]
            if hec1[s1[0]] == hec2[s2[0]]:
                num = sum([s2[0] <= m1[k] < s2[1] for k in range(s1[0], s1[1])])
                overlap1[i] = max(overlap1[i], num / (s1[1] - s1[0]))
                overlap2[j] = max(overlap2[j], num / (s2[1] - s2[0]))
    return (overlap1.mean() + overlap2.mean()) / 2.0

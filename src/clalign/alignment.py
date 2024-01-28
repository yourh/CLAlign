#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

import heapq
import numpy as np
import numba
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

__all__ = ['ProteinSeq', 'AlignmentResult', 'align_core', 'pairwise_align', 'draw_alignment']


@numba.experimental.jitclass({
    'seq': numba.types.unicode_type,
    'pid': numba.types.unicode_type,
    'desc': numba.types.unicode_type,
    'idx': numba.i4
})
class ProteinSeq(object):
    """

    """
    def __init__(self, seq, pid='', desc='', idx=0):
        self.seq, self.pid, self.desc, self.idx = seq.upper(), pid, desc, idx

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, item):
        return self.seq[item]

    def get_record(self):
        return SeqRecord(Seq(self.seq), self.pid)


@numba.experimental.jitclass({
    'seq1': numba.extending.as_numba_type(ProteinSeq),
    'seq2': numba.extending.as_numba_type(ProteinSeq),
    'aln1': numba.types.unicode_type,
    'aln2': numba.types.unicode_type,
    'start1': numba.i4,
    'start2': numba.i4,
    'score': numba.f4,
    'coverage': numba.f4
})
class AlignmentResult(object):
    """

    """

    def __init__(self, seq1, seq2, aln1, aln2=None, start1=0, start2=0, score=0.0, coverage=0.0):
        if aln2 is not None:
            assert len(aln1) == len(aln2)
        else:
            i, j, aln1, aln2, aln_text = 0, 0, '', '', aln1
            for k in range(len(aln_text)):
                match aln_text[k]:
                    case ':' | '.': aln1, aln2, i, j = aln1 + seq1[i], aln2 + seq2[j], i + 1, j + 1
                    case '1': aln1, aln2, i = aln1 + seq1[i], aln2 + '-', i + 1
                    case '2': aln1, aln2, j = aln1 + '-', aln2 + seq2[j], j + 1
                    case _: raise ValueError
            assert i == len(seq1) and j == len(seq2)
        self.seq1, self.seq2, self.aln1, self.aln2, self.start1, self.start2 = seq1, seq2, aln1, aln2, start1, start2
        self.score, self.coverage = score, coverage
        for x, y in zip(self.aln1, self.aln2):
            assert x != '-' or y != '-'
    
    def __lt__(self, other):
        return self.score < other.score

    def __len__(self):
        return len(self.aln1)

    def get_matched_res(self):
        i, j, pos = self.start1, self.start2, []
        for k in range(len(self.aln1)):
            if self.aln1[k] != '-' and self.aln2[k] != '-':
                pos.append((i, j))
            i += self.aln1[k] != '-'
            j += self.aln2[k] != '-'
        return pos


@numba.njit(nogil=True, fastmath=True)
def get_coverage(aln1, aln2, seq_len):
    return len([k for k in range(len(aln1)) if aln1[k] != '-' and aln2[k] != '-']) / seq_len


@numba.njit(nogil=True, fastmath=True)
def empty_protein_seq():
    return ProteinSeq('', '', '', 0)


@numba.njit(nogil=True, fastmath=True)
def empty_aln_res():
    return AlignmentResult(empty_protein_seq(), empty_protein_seq(), '', '', 0, 0, -np.inf, 0.0)


@numba.njit(nogil=True, fastmath=True)
def align_core(seq1: ProteinSeq, seq2: ProteinSeq, sub_mat, gap_penalty=0.0, local=False,
               only_score=False) -> AlignmentResult:
    assert (len(seq1), len(seq2)) == sub_mat.shape
    assert gap_penalty >= 0.0
    f = np.empty((sub_mat.shape[0] + 1, sub_mat.shape[1] + 1), dtype=np.float32)
    g = np.empty_like(f, dtype=np.int32)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            x = np.asarray((0.0 if local or i == j == 0 else -np.inf,
                            f[i - 1, j - 1] + sub_mat[i - 1, j - 1] if i > 0 and j > 0 else -np.inf,
                            f[i - 1, j] - gap_penalty if i > 0 else -np.inf,
                            f[i, j - 1] - gap_penalty if j > 0 else -np.inf))
            g[i, j], f[i, j] = (y := x.argmax()), x[y]
    if local:
        x, y, m = -1, -1, -np.inf
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if f[i, j] > m:
                    m, x, y = f[i, j], i, j
    else:
        x, y = sub_mat.shape
    (i, j), aln1, aln2, s = (x, y), '', '', 0.0
    while i > 0 or j > 0:
        match g[i, j]:
            case 1: aln1, aln2, s = seq1[i := i - 1] + aln1, seq2[j := j - 1] + aln2, s + sub_mat[i, j]
            case 2: aln1, aln2 = seq1[i := i - 1] + aln1, '-' + aln2
            case 3: aln1, aln2 = '-' + aln1, seq2[j := j - 1] + aln2
            case 0: break
    return AlignmentResult(seq1, seq2, aln1 if not only_score else '', aln2 if not only_score else '', i, j,
                           s / (0.5 * (len(seq1) + len(seq2))),
                           get_coverage(aln1, aln2, len(seq1)))


@numba.njit(nogil=True, fastmath=True, parallel=True)
def pairs_align(seqs1: list[ProteinSeq], embs1, seqs2: list[ProteinSeq], embs2, gap_penalty=0.0):
    st1 = np.cumsum(np.asarray([0] + [len(x) for x in seqs1], dtype=np.int32))
    st2 = np.cumsum(np.asarray([0] + [len(x) for x in seqs2], dtype=np.int32))
    aln_res = [empty_aln_res() for _ in range(len(seqs1))]
    for i in numba.prange(len(seqs1)):
        seq1, emb1 = seqs1[i], embs1[st1[i]: st1[i + 1]]
        seq2, emb2 = seqs2[i], embs2[st2[i]: st2[i + 1]]
        aln_res[i] = align_core(seq1, seq2, emb1 @ emb2.T, gap_penalty)
    return aln_res


@numba.njit(nogil=True, fastmath=True, parallel=True)
def pairwise_align(query_seqs: list[ProteinSeq], query_embs, db_seqs: list[ProteinSeq], db_embs, gap_penalty=0.0,
                   local=False, keep=-1, only_score=False, eff_mem=False) -> list[list[AlignmentResult]]:
    query_st = np.cumsum(np.asarray([0] + [len(x) for x in query_seqs], dtype=np.int32))
    db_st = np.cumsum(np.asarray([0] + [len(x) for x in db_seqs], dtype=np.int32))
    keep = keep if keep > 0 else len(db_seqs)
    aln_res = [[empty_aln_res() for _ in range(keep)] for _ in range(len(query_seqs))]
    for i in numba.prange(len(query_seqs)):
        seq1, emb1 = query_seqs[i], query_embs[query_st[i]: query_st[i + 1]]
        res_ = [empty_aln_res() for _ in range(len(db_seqs))]
        for j in range(len(db_seqs)):
            seq2, emb2 = db_seqs[j], db_embs[db_st[j]: db_st[j + 1]]
            res_[j] = align_core(seq1, seq2, emb1 @ emb2.T, gap_penalty, local, only_score or eff_mem)
        for j, r_ in enumerate(res_):
            if keep < len(db_seqs):
                heapq.heappushpop(aln_res[i], r_)
            else:
                aln_res[i][j] = r_
        if not only_score and eff_mem:
            for j in numba.prange(keep):
                seq2 = aln_res[i][j].seq2
                emb2 = db_embs[db_st[seq2.idx]: db_st[seq2.idx + 1]]
                aln_res[i][j] = align_core(seq1, seq2, emb1 @ emb2.T, gap_penalty, local, only_score)
    return aln_res


def draw_alignment(aln_res: AlignmentResult):
    aln = ''.join(['|' if aln_res.aln1[k] != '-' and aln_res.aln2[k] != '-' else ' '
                   for k in range(len(aln_res.aln1))])
    for k in range(0, len(aln), 100):
        print(aln_res.aln1[k: k + 100])
        print(aln[k: k + 100])
        print(aln_res.aln2[k: k + 100])
        print()

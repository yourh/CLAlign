#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

import heapq
import numpy as np
import numba.extending

__all__ = ['ProteinSeq', 'AlignmentResult', 'fast_align', 'pairwise_align', 'draw_alignment']


@numba.experimental.jitclass({
    'seq': numba.types.unicode_type,
    'pid': numba.types.unicode_type
})
class ProteinSeq(object):
    """

    """
    def __init__(self, seq, pid=''):
        self.seq, self.pid = seq.upper(), pid

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, item):
        return self.seq[item]


@numba.experimental.jitclass({
    'seq1': numba.extending.as_numba_type(ProteinSeq),
    'seq2': numba.extending.as_numba_type(ProteinSeq),
    'aln1': numba.types.unicode_type,
    'aln2': numba.types.unicode_type,
    'start1': numba.i4,
    'start2': numba.i4,
    'score': numba.f4
})
class AlignmentResult(object):
    """

    """

    def __init__(self, seq1, seq2, aln1, aln2=None, start1=0, start2=0, score=0.0):
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
        self.score = score
    
    def __lt__(self, other):
        return self.score < other.score

    def get_matched_res(self):
        i, j, pos = self.start1, self.start2, []
        for k in range(len(self.aln1)):
            if self.aln1[k] != '-' and self.aln2[k] != '-':
                pos.append((i, j))
            i += self.aln1[k] != '-'
            j += self.aln2[k] != '-'
        return pos


@numba.njit(nogil=True, fastmath=True)
def fast_align(seq1: ProteinSeq, seq2: ProteinSeq, sub_mat, gap_penalty=0.0, only_score=False) -> AlignmentResult:
    assert (len(seq1), len(seq2)) == sub_mat.shape
    assert gap_penalty >= 0.0
    f = np.empty((sub_mat.shape[0] + 1, sub_mat.shape[1] + 1), dtype=np.float32)
    g = np.empty_like(f, dtype=np.int32)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            x = np.asarray((f[i - 1, j - 1] + sub_mat[i - 1, j - 1] if i > 0 and j > 0 else -np.inf,
                            f[i - 1, j] - gap_penalty if i > 0 else -np.inf,
                            f[i, j - 1] - gap_penalty if j > 0 else -np.inf,
                            0.0 if i == j == 0 else -np.inf))
            g[i, j], f[i, j] = (y := x.argmax()), x[y]
    (i, j), aln1, aln2, s = sub_mat.shape, '', '', 0.0
    while i > 0 or j > 0:
        match g[i, j]:
            case 0: aln1, aln2, s = seq1[i := i - 1] + aln1, seq2[j := j - 1] + aln2, s + sub_mat[i, j]
            case 1: aln1, aln2 = seq1[i := i - 1] + aln1, '-' + aln2
            case 2: aln1, aln2 = '-' + aln1, seq2[j := j - 1] + aln2
    return AlignmentResult(seq1, seq2, aln1 if not only_score else '', aln2 if not only_score else '', 0, 0,
                           s / (len(seq1) - aln2.count('-')))


@numba.njit(nogil=True, fastmath=True, parallel=True)
def pairwise_align(query_seqs: list[ProteinSeq], query_embs, db_seqs: list[ProteinSeq], db_embs, keep,
                   gap_penalty=0.0, only_score=False):
    query_st = np.cumsum(np.asarray([0] + [len(x) for x in query_seqs], dtype=np.int32))
    db_st = np.cumsum(np.asarray([0] + [len(x) for x in db_seqs], dtype=np.int32))
    aln_res = [[AlignmentResult(ProteinSeq('', ''), ProteinSeq('', ''), '', '', 0, 0, -np.inf) for _ in range(keep)]
               for _ in range(len(query_seqs))]
    for i in numba.prange(len(query_seqs)):
        seq1, emb1 = query_seqs[i], query_embs[query_st[i]: query_st[i + 1]]
        res_ = [AlignmentResult(ProteinSeq('', ''), ProteinSeq('', ''), '', '', 0, 0, 0) for _ in range(len(db_seqs))]
        for j in numba.prange(len(db_seqs)):
            seq2, emb2 = db_seqs[j], db_embs[db_st[j]: db_st[j + 1]]
            res_[j] = fast_align(seq1, seq2, emb1 @ emb2.T, gap_penalty, only_score)
        for r_ in res_:
            heapq.heappushpop(aln_res[i], r_)
    return aln_res


def draw_alignment(aln_res: AlignmentResult):
    aln = ''.join(['|' if aln_res.aln1[k] != '-' and aln_res.aln2[k] != '-' else ' '
                   for k in range(len(aln_res.aln1))])
    for k in range(0, len(aln), 100):
        print(aln_res.aln1[k: k + 100])
        print(aln[k: k + 100])
        print(aln_res.aln2[k: k + 100])
        print()

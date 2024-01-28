# Available Baseline Results

Inventory of the existing research results, checked 2026-10-16. This is a file
coverage audit, not a rerun of any baseline. The tables below describe locally
available results, not everything included in this repository. Selected
MALIDUP/MALISAM metric tables, SCOPe `.rocx` tables, and frozen plotting inputs
are in `results/`. Benchmark references are reused from the repository's `data/`
directory. Individual MALI alignment files and raw SCOPe score tables are
retained locally for possible reviewer requests.

## MALIDUP And MALISAM

| Method | MALIDUP (241 pairs) | MALISAM (130 pairs) | Available detail |
| --- | --- | --- | --- |
| pLM-BLAST | 241 | 130 | Pair alignments and metric tables |
| PLMAlign | 241 | 130 | Pair alignments and metric tables |
| EBA | 241 | 130 | Pair alignments and metric tables |
| DeepBLAST | 241 | 130 | Pair alignments and metric tables |
| TEA-NW | 241 | 130 | Pair alignments and metric tables |
| TEA (alternative variant) | 241 | 130 | Pair alignments and metric tables |
| FAST | 241 | 130 | Dataset-supplied `.fast.ali` files and metric tables |
| TM-align | 241 | 130 | Pair alignments and metric tables |
| Dali | 234 successes + 7 failures | 130 | 234/130 pair alignments; 241/130 metric rows |
| GTAlign | 241 | 130 | Pair alignments and metric tables |

Metric tables: `results/{malidup,malisam}/{method}.txt` in the original research
workspace. Most per-pair files: `results/{dataset}/{method}/{pair}.txt`.
FAST originals: `data/{dataset}/{pair}/*.fast.ali`. TEA variants also have per-pair
files under the release workspace's `.conda/experiments/tea-malidata/results*/`.

Manual reference alignments and all three CLAlign encoder variants (ProtT5,
ProstT5, ESM-IF) have complete 241/130 per-pair files and metric rows. Older
`clalign` results and `clalign_sa.txt` metric tables also exist; they must not be
mixed with the chosen encoder-specific results without checking provenance.

No complete NW, SW, DEDAL or Mammoth-local prediction set was located in the
inspected workspaces. A DEDAL example notebook is not a dataset-level result set.

## SCOPe40-Test

| Method | Local source score rows | Queries | Targets | Per-query curve rows |
| --- | ---: | ---: | ---: | ---: |
| pLM-BLAST | 4,870,849 | 2,207 | 2,207 | 766 |
| PLMAlign | 4,870,849 | 2,207 | 2,207 | 766 |
| EBA | 4,870,849 | 2,207 | 2,207 | 766 |
| TEA | 4,870,849 | 2,207 | 2,207 | 766 |
| TM-align | 4,870,849 | 2,207 | 2,207 | 766 |
| GTAlign | 4,870,849 | 2,207 | 2,207 | 766 |
| CLAlign-ProtT5 | 4,870,849 | 2,207 | 2,207 | 766 |

Files: `results/scope40201/{method}.txt` and `{method}.rocx` in the research
workspace. Every audited `.txt` file has exactly 2,207 targets per query, 2,207
self-hits, no duplicate query-target pairs, and no non-finite scores. These are
`query target score` tables, not residue-level alignment strings. The `.rocx`
files all cover the same 766 eligible queries.

Additional CLAlign-ProstT5 and CLAlign-ESMIF `.txt`, `.csv` and `.rocx` files
exist, as do older CLAlign variants; their large score tables are not included
in the seven-file all-pairs audit above. No SCOPe prediction set was located for
DeepBLAST, FAST or Dali. Their absence is not a missing row in the six-baseline
SCOPe comparison used by these figures.

## Repository Contents

The available results cover all methods used by these plotting scripts, including
the explicitly recorded Dali failures. They do not constitute a complete raw
prediction archive for every baseline mentioned in the manuscript. This directory
contains the selected variants, keeps failures explicit, and
omits individual MALIDUP/MALISAM alignment files, raw SCOPe all-versus-all score
tables, and unneeded multi-gigabyte intermediates. SCOPe `.rocx` files and the
fixed retrieval-metric values are sufficient for the provided figures.
All selected MALI method-level metric tables are retained. `results/inventory.json`
records source and included file coverage; `results/manifest.json` records checksums
with repository-relative paths. Each SCOPe curve table is stored once, and
reference sequences and annotations are reused from the repository's `data/`
directory.

The EBA case-study heatmap uses author-original T5 residue extraction. Historical
EBA benchmark files are preserved as found. Current source state alone cannot
establish past execution provenance; the separate rerun audit tests whether the
saved results can be reproduced with the author-original implementation.

Author-original EBA was subsequently rerun on all 241 MALIDUP and 130 MALISAM
pairs: every saved alignment string, F1, HEC ACC and HEC SOV value was reproduced
exactly. A SCOPe pilot covered 24 queries against all 2,207 targets (52,968 pairs);
all scores were within tolerance, with maximum absolute difference 2.861e-6 and
identical ordered top-1/top-10 lists for all 24 queries. SCOPe was not numerically
rerun in full. Detailed verification evidence is retained locally and is not
included in this directory.

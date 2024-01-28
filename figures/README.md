# CLAlign Manuscript Figures

Plotting scripts, frozen experimental results, and 19 PDF figures.

## Contents

- `results/`: metric tables, SCOPe sensitivity tables, and frozen plotting inputs.
- `output/`: generated PDF figures.

## Generate Figures

Use Python 3.10-3.12. No GPU or model downloads are required.

Run from the repository root:

```bash
python -m pip install -r figures/requirements.txt
python figures/scatter.py       # TM-score plots
python figures/hec.py           # ACC and SOV
python figures/roc.py           # SCOPe sensitivity curves
python figures/retrieval.py     # MAP, P@1 and P@10
python figures/gap_penalty.py   # Gap-penalty analysis
python figures/heatmap.py       # Similarity heatmaps
```

PDFs are saved to `figures/output/`.

## Data Notes

- MALIDUP/MALISAM metric columns: `pair_id precision recall F1 TM_score [HEC_ACC HEC_SOV]`.
- HEC averages use 240 valid MALIDUP pairs and 129 valid MALISAM pairs. Seven MALIDUP Dali failures are recorded in `results/failures.tsv`.
- SCOPe curves use `.rocx` tables; retrieval bar charts use fixed reported values in `retrieval.py`.
- Benchmark references are the repository's `data/malidup_hec.csv`, `data/malisam_hec.csv`, and `data/scope40-test.fasta`.

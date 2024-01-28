CLAlign
=======

CLAlign aligns protein sequences or protein structures with residue-level
embeddings from protein language models.

Installation
------------

CLAlign currently targets Python 3.12+ and CUDA 12.8 builds of PyTorch.
Install from the repository root with:

```bash
pip install . \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://urob.github.io/numpy-mkl \
  -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
```

The package includes the default LoRA adapters for:

- `CLAlign-ProtT5`
- `CLAlign-ProstT5`
- `CLAlign-ESMIF`

Pairwise Alignment
------------------

For a quick pairwise alignment, create a text file containing two protein
sequences, one per line. This example uses the MALISAM pair
`d1cs1a_d1q8ia1`:

```text
AGAVLTNTGMSAIHLVTTVFLDLVLHSCTKYLNGHSDVVAGVVIAKDPDVVTELAWWANNIGVTG
QESVAFIPADQVPRAQHILQGEQGFRLTPLALKDFHRQPVYGLYCRAHRQLMNYEKRLREGGVTVYEADVR
```

Then run:

```bash
clalign align pair pair.txt
```

Example output:

```text
Alignment Score: 0.282163
AGA-V-L-TNT-GMSAIHLVTTVFLDLVLHSCTKYLNGHSDVVAGVVIAKDPDVVTELAWWANNIGVT------G
 || | | ||| |||||||||||||||||| |||| | ||||||||||||||||||||||||||||||      |
-QESVAFIPADQVPRAQHILQGEQGFRLTP-LALK-D-FHRQPVYGLYCRAHRQLMNYEKRLREGGVTVYEADVR
```

Batch Alignment
---------------

Align query proteins against reference proteins:

```bash
clalign align query query.fasta reference.fasta results.csv
```

The output is a CSV file with query ID, reference ID, aligned strings, start
positions, and alignment score. By default, CLAlign reports hits with at least
20% query coverage. Change this cutoff with `--threshold`.

For local alignment:

```bash
clalign align query query.fasta reference.fasta results.csv --local
```

For large searches, generate embeddings once and reuse them:

```bash
clalign generate-embs query.fasta query.clalign
clalign generate-embs reference.fasta reference.clalign

clalign --use-precomputed align query \
  query.fasta reference.fasta results.csv \
  --qe query.clalign \
  --re reference.clalign
```

If an embedding path without `.npy` is provided, CLAlign will also try the same
path with `.npy` appended.

Remote Homology Search
----------------------

CLAlign can be used as an embedding-based remote homology search tool by
aligning a query FASTA against a reference database FASTA:

```bash
clalign align query query.fasta reference.fasta homology_hits.csv \
  --only-score \
  --threshold 0.2 \
  --keep 100
```

`--keep` controls how many top reference hits are retained per query. Values
between 0 and 1 are interpreted as a fraction of the reference set; values
greater than or equal to 1 are interpreted as an absolute hit count.

Options
-------

By default, CLAlign loads the bundled `CLAlign-ProtT5` LoRA adapter on top of
`Rostlab/prot_t5_xl_uniref50`.

Common options:

```bash
clalign --help
clalign align query --help
```

Useful flags include:

- `--model-path`: path to a LoRA adapter or model directory
- `--lora/--no-lora`: enable or disable LoRA loading
- `--base-model-name`: base protein language model
- `--max-length`: maximum model input length
- `--batch-size`: embedding batch size
- `--device`: device such as `cuda` or `cpu`
- `--amp`: enable autocast mixed precision
- `--pdb-dir`: read structure inputs from PDB files listed by name
- `--local`: use local alignment
- `--only-score`: skip storing alignment strings
- `--threshold`: minimum query coverage for reported hits
- `--use-precomputed`: require `--qe`/`--re` embeddings instead of loading a model

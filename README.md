# Install
```bash
git clone https://github.com/yourh/CLAlign
cd CLAlign
pip install . --extra-index-url https://download.pytorch.org/whl/cu124
```

# Align a pair of proteins
```bash
clalign --lora --max-length 2048 -a align pair pair.txt
```

# Align pairwise proteins

```bash
export query_fasta=data/scope40-2.01-test.fasta
export db_fasta=data/scope40-2.01-test.fasta
export db_embs=data/scope40-2.01-test-claligndb.npy
export res=results/scope40201/clalign.csv

clalign \
--lora \
--max-length 2048 \
-a \
create-db \
${db_fasta} \
${db_embs}

# "--only-score" option allows CLAlign output only alignment scores 
OMP_NUM_THREADS=1 clalign \
--lora \
--max-length 2048 \
-a \
align \
-g 0.0 \
query \
${query_fasta} \
${db_fasta} \
${res} \
--de ${db_embs} \
--only-score
```

# Train CLAlign
```bash
torchrun \
--nproc-per-node 4 \
--no-python \
clalign \
--lora \
--max-length 512 \
--model-path CLAlignT5 \
-a \
cl \
--train-data data/train10k.txt \
-e 3 \
-b 2 \
--dist \
-w 0.1
```


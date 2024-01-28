"""Optional one-pair inference; NOT needed to run heatmap.py or other plots."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results"
HALF_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
FULL_MODEL = "Rostlab/prot_t5_xl_uniref50"


def residue_states(hidden, sequence_length):
    """T5 has no leading CLS token: retain all residues and omit terminal EOS."""
    if hidden.shape[0] != sequence_length + 1:
        raise ValueError("Expected residue states followed by exactly one EOS state")
    return hidden[:sequence_length]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Local CLAlign-ProtT5 adapter directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DATA)
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow model downloads; otherwise use the local cache only")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    local_only = not args.allow_download
    case = json.loads((DATA / "heatmap_case.json").read_text())
    sequences = [case["sequence1"], case["sequence2"]]
    device = torch.device(args.device)

    def embeddings(model, tokenizer, sequence, keep_device=False, include_eos=False):
        tokens = tokenizer([" ".join(sequence.upper())], return_tensors="pt")
        if tokens["input_ids"].shape[1] != len(sequence) + 1:
            raise ValueError("Expected one token per residue and one terminal EOS")
        ids = tokens["input_ids"][0].tolist()
        if ids[-1] != tokenizer.eos_token_id or tokenizer.eos_token_id in ids[:-1]:
            raise ValueError("Expected EOS only at the end of the token sequence")
        with torch.inference_mode():
            hidden = model(**{k: v.to(device) for k, v in tokens.items()}).last_hidden_state[0]
        residues = hidden if include_eos else residue_states(hidden, len(sequence))
        return residues if keep_device else residues.cpu()

    def zscore(matrix):
        return (matrix - matrix.mean()) / (matrix.std() + 1e-8)

    print("Computing pLM-BLAST / PLMAlign matrices for one pair", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(HALF_MODEL, use_fast=False, local_files_only=local_only)
    model = T5EncoderModel.from_pretrained(HALF_MODEL, torch_dtype=torch.float32,
                                          local_files_only=local_only).to(device).eval()
    half_revision = getattr(model.config, "_commit_hash", None)
    first, second = [embeddings(model, tokenizer, seq) for seq in sequences]
    matrices = {
        "plmalign": zscore(first @ second.T).numpy(),
        "plmblast": zscore(functional.normalize(first) @ functional.normalize(second).T).numpy(),
    }
    print("Computing CLAlign matrix for the same pair", flush=True)
    adapted = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    first, second = [embeddings(adapted, tokenizer, seq) for seq in sequences]
    matrices["clalign"] = (functional.normalize(first) @ functional.normalize(second).T).numpy()
    del adapted, model, first, second
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Computing EBA matrix with author-original residue extraction", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(FULL_MODEL, do_lower_case=False, local_files_only=local_only)
    model = T5EncoderModel.from_pretrained(FULL_MODEL, torch_dtype=torch.float32,
                                          local_files_only=local_only).to(device).eval()
    full_revision = getattr(model.config, "_commit_hash", None)
    first, second = [embeddings(model, tokenizer, seq, keep_device=True, include_eos=True)
                     for seq in sequences]
    # Preserve EBA's author-original slicing, including the terminal EOS token.
    similarity = torch.exp(-torch.cdist(first[1:], second[1:], p=2))
    row_mean = similarity.sum(1) / similarity.shape[1]
    column_mean = similarity.sum(0) / similarity.shape[0]
    z_rows = (similarity - row_mean[:, None]) / similarity.std(1)[:, None]
    z_columns = (similarity - column_mean) / similarity.std(0)
    matrices["eba"] = ((z_rows + z_columns) / 2).cpu().numpy()

    expected_shape = tuple(map(len, sequences))
    for name, matrix in matrices.items():
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {name} matrix")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "heatmap_matrices.npz"
    np.savez_compressed(destination, **matrices)
    adapter = args.adapter / "adapter_model.safetensors"
    metadata = {
        "pair_id": case["pair_id"], "shape": expected_shape,
        "models": {"plmblast_plmalign_clalign": HALF_MODEL, "eba": FULL_MODEL},
        "model_revisions": {HALF_MODEL: half_revision, FULL_MODEL: full_revision},
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "matrices_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "precision": "float32; eval mode; seed 0; TF32 disabled",
        "device": str(device),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "torch", "transformers", "peft", "sentencepiece"]},
        "formulas": {
            "plmblast": "global sample z-score of cosine similarity; epsilon=1e-8",
            "plmalign": "global sample z-score of raw dot product; epsilon=1e-8",
            "clalign": "cosine similarity of adapter embeddings; no z-score",
            "eba": "mean of row and column sample z-scores of exp(-Euclidean distance)",
        },
        "eba_default": "author-original (array key: eba)",
        "token_conventions": {
            "plmblast_plmalign_clalign": "hidden[0, :-1]: all residues, excluding terminal EOS",
            "eba": "hidden[0, 1:L+1]: author-original; drops first residue and includes terminal EOS",
        },
        "note": "Freshly computed for the case named by the notebooks; not recovered historical arrays. "
                "The saved EBA notebook output refers to a different pair and is stale. "
                "EBA uses the author-original convention. Historical benchmark files are unchanged. "
                "Current code state alone cannot establish historical execution provenance; see the separate rerun audit.",
    }
    (args.output_dir / "heatmap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(destination, flush=True)


if __name__ == "__main__":
    main()

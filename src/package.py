"""Assemble the Hugging Face model folder and optionally upload it (as a private repo).

    python -m src.package                                    # -> outputs/hub
    python -m src.package --push-to-hub zackyalgiffari/ott-recommender-ml32m

Folder contents: config.json + model.safetensors (weights), item_content.safetensors (frozen
text embeddings), catalog.parquet (titles only, no reviews or plot text), modeling.py,
recommend.py, README.md (from MODEL_CARD.md), requirements.txt and results/*.json.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import polars as pl

from src.config import DOMAINS, RESULTS_DIR, ROOT, processed_dir
from src.data import Sequences

HUB_REQUIREMENTS = "torch>=2.1\nhuggingface_hub>=0.30\nsafetensors>=0.4\npandas>=2.0\npyarrow>=15\ntransformers>=4.46\n"


def build(model_dir: Path, processed: Path, out: Path, reset_cold_ids: bool = False) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "results").mkdir(parents=True)
    if reset_cold_ids:
        import numpy as np
        import torch

        from src.model import HybridSASRec

        model = HybridSASRec.from_pretrained(str(model_dir))
        counts = Sequences.load(processed).train_counts()
        n = model.reset_cold_ids(torch.from_numpy(np.flatnonzero(counts == 0)))
        model.save_pretrained(out)
        (out / "README.md").unlink(missing_ok=True)  # mixin's stub card; MODEL_CARD.md is copied below
        print(f"[package] zeroed the ID vectors of {n:,} titles with no training events", flush=True)
    else:
        for f in ("config.json", "model.safetensors"):
            shutil.copy2(model_dir / f, out / f)
    content = processed / "item_content.safetensors"
    if content.exists():
        shutil.copy2(content, out / "item_content.safetensors")

    seq = Sequences.load(processed)
    counts = seq.train_counts()
    items = pl.read_parquet(processed / "items.parquet")
    catalog = items.select(
        "item_idx",
        pl.col("domain").map_elements(lambda d: DOMAINS[d], return_dtype=pl.Utf8).alias("domain"),
        "source_id", "title", "year", "genres", "fmt", "has_title", "tmdb_id",
        pl.Series("popularity", counts[items["item_idx"].to_numpy()]).alias("popularity"),
    )
    catalog.write_parquet(out / "catalog.parquet")

    shutil.copy2(ROOT / "src" / "model.py", out / "modeling.py")
    shutil.copy2(ROOT / "src" / "recommend.py", out / "recommend.py")
    card = ROOT / "MODEL_CARD.md"
    if card.exists():
        shutil.copy2(card, out / "README.md")
    (out / "requirements.txt").write_text(HUB_REQUIREMENTS)
    for f in ("metrics.json", "data_report.json", "train_history.json", "ablations.json",
              "results_ndcg.png", "history_length.png"):
        if (RESULTS_DIR / f).exists():
            shutil.copy2(RESULTS_DIR / f, out / "results" / f)
    return out


def push(folder: Path, repo_id: str, message: str, revision: str | None = None) -> None:
    from huggingface_hub import HfApi

    token = None
    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path and Path(token_path).exists():
        token = Path(token_path).read_text().strip()
    api = HfApi(token=token)
    # Created private; the visibility is changed separately once the release is reviewed.
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if revision:  # candidate models go to a branch, never over main
        api.create_branch(repo_id, branch=revision, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type="model", commit_message=message,
                      revision=revision)
    print(f"[hub] pushed to https://huggingface.co/{repo_id}" + (f" (branch {revision})" if revision else ""),
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=Path("outputs/best"))
    p.add_argument("--sample-users", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/hub"))
    p.add_argument("--push-to-hub", metavar="REPO_ID")
    p.add_argument("--revision", default=None, help="push to this branch instead of main")
    p.add_argument("--reset-cold-ids", action="store_true",
                   help="zero the ID vectors of titles with no training events (content-only scoring)")
    p.add_argument("--message", default="Upload OTT hybrid recommender")
    args = p.parse_args()

    out = build(args.model_dir, processed_dir(args.sample_users), args.out, args.reset_cold_ids)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"[package] {out} ({size:.0f} MB): " + ", ".join(sorted(f.name for f in out.iterdir())), flush=True)
    if args.push_to_hub:
        push(out, args.push_to_hub, args.message, args.revision)


if __name__ == "__main__":
    main()

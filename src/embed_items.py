"""Embed every catalog item's metadata text with a frozen Hugging Face sentence encoder.

    python -m src.embed_items                                   # BAAI/bge-base-en-v1.5
    python -m src.embed_items --sample-users 500 --encoder BAAI/bge-small-en-v1.5 --device cpu

Writes ``item_content.safetensors`` (tensor "content", float16, shape (n_items + 1, dim),
row 0 = padding) next to items.parquet. BGE uses CLS pooling + L2 normalisation.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

from src.config import ENCODER, processed_dir


@torch.no_grad()
def encode(texts: list[str], encoder: str, device: torch.device, batch_size: int = 256,
           max_tokens: int = 256, log: bool = True) -> torch.Tensor:
    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    order = np.argsort([len(t) for t in texts])[::-1]  # long first: fail fast on memory
    out = torch.zeros(len(texts), model.config.hidden_size, dtype=torch.float16)
    t0 = time.time()
    for b, i in enumerate(range(0, len(texts), batch_size)):
        idx = order[i : i + batch_size]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_tokens,
                  return_tensors="pt").to(device)
        cls = model(**enc).last_hidden_state[:, 0].float()
        out[torch.from_numpy(idx.copy())] = torch.nn.functional.normalize(cls, dim=-1).half().cpu()
        if log and b % 100 == 0:
            print(f"[embed] {i + len(idx):,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default=ENCODER)
    p.add_argument("--sample-users", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    d = processed_dir(args.sample_users)
    items = pl.read_parquet(d / "items.parquet").sort("item_idx")
    if items["item_idx"].to_list() != list(range(1, items.height + 1)):
        raise RuntimeError("item_idx must be 1..N without gaps")
    emb = encode(items["text"].to_list(), args.encoder, torch.device(args.device), args.batch_size, args.max_tokens)
    content = torch.cat([torch.zeros(1, emb.shape[1], dtype=torch.float16), emb])
    save_file({"content": content}, str(d / "item_content.safetensors"),
              metadata={"encoder": args.encoder, "pooling": "cls", "normalized": "true",
                        "max_tokens": str(args.max_tokens)})
    print(f"[embed] wrote {d / 'item_content.safetensors'} shape={tuple(content.shape)}")


if __name__ == "__main__":
    main()

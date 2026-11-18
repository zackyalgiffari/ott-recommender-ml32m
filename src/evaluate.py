"""Full-catalog evaluation of a trained model, plus a popularity baseline and a content-only mode.

    python -m src.evaluate --model outputs/best --out results/metrics.json

For every domain and every test user, the held-out last title is ranked against the whole
catalog of that domain (already-watched titles excluded). Reported variants:
    hybrid        the model as trained (ID + content item vectors)
    content_only  ID vectors zeroed -> how the model ranks titles it has never seen (cold start)
    popularity    training-set popularity, the standard non-personalised baseline
Plus the "cold" slice: test targets that never occur in any training sequence.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from src.config import DOMAINS, RESULTS_DIR, processed_dir
from src.data import Sequences, eval_inputs, history_index
from src.metrics import RankAccumulator
from src.model import HybridSASRec

DEFAULT_MAX_LEN = {"movielens": 200, "amazon": 50}


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def rank_domain(
    seq: Sequences,
    domain: int,
    split: str,
    users: np.ndarray,
    device: torch.device,
    model: HybridSASRec | None = None,
    max_len: int = 200,
    id_scale: float = 1.0,
    pop_counts: np.ndarray | None = None,
    batch_users: int = 1024,
) -> RankAccumulator:
    """Rank the held-out item of ``users`` against the full catalog of ``domain``."""
    lo, hi = seq.meta["domain_ranges"][domain]
    acc = RankAccumulator(n_items=hi - lo)
    if model is not None:
        model.eval()
        with autocast(device):
            item_mat = model.domain_items(domain, id_scale=id_scale).float()
    else:
        pop = torch.as_tensor(pop_counts[lo:hi], dtype=torch.float32, device=device)
    dom = torch.tensor(domain, device=device)
    for i in range(0, len(users), batch_users):
        chunk = users[i : i + batch_users]
        inp, tgt, first, end = eval_inputs(seq, chunk, split, max_len)
        if model is not None:
            with autocast(device):
                h = model.user_state(torch.from_numpy(inp).to(device), dom.expand(len(chunk)), id_scale=id_scale)
            scores = h.float() @ item_mat.T
        else:
            scores = pop[None, :].repeat(len(chunk), 1)
        rows, cols = history_index(seq.items, first, end, lo)
        scores[torch.from_numpy(rows).to(device), torch.from_numpy(cols).to(device)] = float("-inf")
        acc.add(scores, torch.from_numpy(tgt - lo).to(device))
    return acc


def load_model(model_dir: Path, processed: Path, device: torch.device) -> HybridSASRec:
    model = HybridSASRec.from_pretrained(str(model_dir))
    if model.use_content:
        model.set_content(load_file(str(processed / "item_content.safetensors"))["content"])
    return model.to(device).eval()


def evaluate_all(model, seq, device, max_len: dict, split: str = "test", domains=None,
                 content_only: bool = True, popularity: bool = True) -> dict:
    counts = seq.train_counts()
    out = {}
    for d, name in enumerate(DOMAINS):
        if domains is not None and name not in domains:
            continue
        users = seq.users(d)
        _, tgt, _, _ = eval_inputs(seq, users, split, 1)
        cold = torch.from_numpy(counts[tgt] == 0)
        res = {}
        t0 = time.time()
        variants = [("hybrid", dict(model=model, id_scale=1.0))]
        if content_only and model.use_content and model.use_id:
            variants.append(("content_only", dict(model=model, id_scale=0.0)))
        if popularity:
            variants.append(("popularity", dict(pop_counts=counts)))
        for vname, kw in variants:
            acc = rank_domain(seq, d, split, users, device, max_len=max_len[name], **kw)
            res[vname] = acc.result()
            res[vname]["cold_targets"] = acc.result(mask=cold)
        if not model.use_content or not model.use_id:
            res["model_variant"] = "content-only model" if model.use_content else "ID-only model"
        print(f"[eval] {split} {name}: {len(users):,} users in {time.time() - t0:.0f}s "
              + " | ".join(f"{v} NDCG@10={res[v]['ndcg@10']:.4f} R@10={res[v]['recall@10']:.4f}"
                           for v, _ in variants), flush=True)
        out[name] = res
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("outputs/best"))
    p.add_argument("--sample-users", type=int, default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "metrics.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    processed = processed_dir(args.sample_users)
    seq = Sequences.load(processed)
    model = load_model(args.model, processed, device)
    max_len = {"movielens": min(DEFAULT_MAX_LEN["movielens"], model.max_len),
               "amazon": min(DEFAULT_MAX_LEN["amazon"], model.max_len)}
    metrics = {
        "protocol": "full-catalog ranking per domain, leave-last-out, already-watched titles excluded; "
                    "ties broken by item index",
        "split": args.split,
        "split_sha256": seq.meta["split_sha256"],
        "model_config": {k: v for k, v in model._hub_mixin_config.items() if k != "domain_ranges"}
        if hasattr(model, "_hub_mixin_config") else {},
        "results": evaluate_all(model, seq, device, max_len, args.split, domains=args.domains.split(",")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()

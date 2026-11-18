"""Train HybridSASRec on the prepared two-domain watch sequences.

    python -m src.train                                  # full run (GPU)
    python -m src.train --no-content --output-dir outputs/ablation-id-only
    python -m src.train --domains movielens --output-dir outputs/ablation-ml-only
    python -m src.train --sample-users 500 --epochs 1 --device cpu   # smoke

Every batch holds windows of a single domain, so the sampled negatives come from that
domain's catalog. Each epoch visits every training window once (domains interleaved at
random, i.e. proportional mixing). The best epoch by mean validation NDCG@10 is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from src.config import DOMAINS, ENCODER, RESULTS_DIR, processed_dir
from src.data import Sequences, train_batch, train_windows
from src.evaluate import autocast, rank_domain
from src.model import HybridSASRec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-users", type=int, default=None, help="use the sampled dataset from prepare.py")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--history-out", type=Path, default=RESULTS_DIR / "train_history.json")
    p.add_argument("--domains", default=",".join(DOMAINS), help="comma-separated subset of domains to train on")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--id-dropout", type=float, default=0.2)
    p.add_argument("--no-content", action="store_true", help="ID-only SASRec ablation")
    p.add_argument("--no-id", action="store_true", help="content-only ablation")
    p.add_argument("--max-len-ml", type=int, default=200)
    p.add_argument("--max-len-az", type=int, default=50)
    p.add_argument("--tokens-per-batch", type=int, default=25_600)
    p.add_argument("--negatives", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="cap (smoke tests)")
    p.add_argument("--val-users", type=int, default=20_000, help="validation users per domain per epoch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


def sampled_softmax_loss(model, inp, tgt, domain: int, lo: int, hi: int, n_neg: int) -> torch.Tensor:
    """Cross-entropy of the true next item against ``n_neg`` uniform same-domain negatives."""
    dom = torch.full((inp.shape[0],), domain, device=inp.device)
    h = model(inp, dom)
    m = (inp != 0) & (tgt != 0)
    h, t = h[m], tgt[m]
    neg = torch.randint(lo, hi, (n_neg,), device=inp.device)
    v_pos = model.item_vectors(t, drop_id=True)
    v_neg = model.item_vectors(neg, drop_id=True)
    pos_logit = (h * v_pos).sum(-1, keepdim=True)
    neg_logit = (h @ v_neg.T).masked_fill(neg[None, :] == t[:, None], float("-inf"))
    logits = torch.cat([pos_logit, neg_logit], dim=1).float()
    return F.cross_entropy(logits, torch.zeros(len(t), dtype=torch.long, device=inp.device))


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (no_decay if prm.ndim < 2 or "emb" in name else decay).append(prm)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    processed = processed_dir(args.sample_users)
    seq = Sequences.load(processed)
    meta = seq.meta
    domains = [DOMAINS.index(d) for d in args.domains.split(",")]
    max_len = {0: args.max_len_ml, 1: args.max_len_az}

    use_content, use_id = not args.no_content, not args.no_id
    content = None
    content_dim, encoder = 768, ENCODER
    if use_content:
        f = processed / "item_content.safetensors"
        content = load_file(str(f))["content"]
        content_dim = content.shape[1]
        from safetensors import safe_open
        with safe_open(str(f), "pt") as sf:
            encoder = (sf.metadata() or {}).get("encoder", ENCODER)

    model = HybridSASRec(
        n_items=meta["n_items"], content_dim=content_dim, d_model=args.d_model, n_layers=args.layers,
        n_heads=args.heads, max_len=max(max_len.values()), n_domains=len(DOMAINS), dropout=args.dropout,
        id_dropout=args.id_dropout if (use_id and use_content) else 0.0, use_id=use_id, use_content=use_content,
        domains=list(DOMAINS), domain_ranges=meta["domain_ranges"], encoder=encoder,
    )
    if content is not None:
        model.set_content(content)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params:,} device={device} domains={[DOMAINS[d] for d in domains]} "
          f"use_id={use_id} use_content={use_content} encoder={encoder}", flush=True)

    windows = {d: train_windows(seq, d, max_len[d]) for d in domains}
    bsz = {d: max(1, args.tokens_per_batch // max_len[d]) for d in domains}
    steps_per_epoch = sum(math.ceil(len(windows[d][0]) / bsz[d]) for d in domains)
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    for d in domains:
        print(f"[train] {DOMAINS[d]}: {len(windows[d][0]):,} windows, batch={bsz[d]}, max_len={max_len[d]}", flush=True)
    print(f"[train] {steps_per_epoch:,} steps/epoch, up to {args.epochs} epochs", flush=True)

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.98))
    warm = min(args.warmup_steps, max(1, total_steps // 10))

    def lr_at(step: int) -> float:  # linear warm-up, cosine decay to 10%
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    rng = np.random.default_rng(args.seed)
    val_users = {}
    for d in domains:
        u = seq.users(d)
        val_users[d] = np.sort(rng.choice(u, size=min(args.val_users, len(u)), replace=False))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    history, best, bad, start_epoch, step = [], -1.0, 0, 0, 0
    ckpt = out / "last.pt"
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        history, best, bad, start_epoch, step = (state[k] for k in ("history", "best", "bad", "epoch", "step"))
        rng = state["rng"]
        print(f"[train] resumed after epoch {start_epoch} (best={best:.4f})", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        plan = []
        for d in domains:
            starts, ends = windows[d]
            perm = rng.permutation(len(starts))
            plan += [(d, perm[i : i + bsz[d]]) for i in range(0, len(perm), bsz[d])]
        rng.shuffle(plan)
        plan = plan[:steps_per_epoch]
        t0, run_loss, n_loss = time.time(), 0.0, 0
        for i, (d, idx) in enumerate(plan):
            starts, ends = windows[d]
            inp, tgt = train_batch(seq, starts[idx], ends[idx])
            inp = torch.from_numpy(inp).to(device, non_blocking=True)
            tgt = torch.from_numpy(tgt).to(device, non_blocking=True)
            lo, hi = meta["domain_ranges"][d]
            with autocast(device):
                loss = sampled_softmax_loss(model, inp, tgt, d, lo, hi, args.negatives)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            n_loss += 1
            if (i + 1) % args.log_every == 0:
                el = time.time() - t0
                print(f"[train] epoch {epoch + 1} step {i + 1}/{len(plan)} loss={run_loss / n_loss:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {(i + 1) / el:.1f} it/s", flush=True)
        train_time = time.time() - t0

        t1 = time.time()
        val = {}
        for d in domains:
            acc = rank_domain(seq, d, "val", val_users[d], device, model=model, max_len=max_len[d])
            val[DOMAINS[d]] = acc.result()
        score = float(np.mean([v["ndcg@10"] for v in val.values()]))
        rec = {"epoch": epoch + 1, "loss": round(run_loss / max(1, n_loss), 5), "train_seconds": round(train_time),
               "val_seconds": round(time.time() - t1), "val_mean_ndcg@10": round(score, 5), "val": val}
        history.append(rec)
        print(f"[train] epoch {epoch + 1} loss={rec['loss']:.4f} val_mean_ndcg@10={score:.4f} "
              + " ".join(f"{k}:NDCG@10={v['ndcg@10']:.4f},R@10={v['recall@10']:.4f}" for k, v in val.items())
              + f" ({train_time:.0f}s train)", flush=True)
        if score > best:
            best, bad = score, 0
            model.save_pretrained(out / "best")
            print(f"[train] new best -> {out / 'best'}", flush=True)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "history": history, "best": best, "bad": bad, "epoch": epoch + 1, "step": step, "rng": rng}, ckpt)
        args.history_out.parent.mkdir(parents=True, exist_ok=True)
        args.history_out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                "params": n_params, "history": history}, indent=2))
        if bad >= args.patience:
            print(f"[train] early stop: no improvement for {args.patience} epochs", flush=True)
            break
    print(f"[train] done. best val_mean_ndcg@10={best:.4f}", flush=True)


if __name__ == "__main__":
    main()

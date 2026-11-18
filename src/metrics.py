"""Full-catalog ranking metrics (no sampled candidates).

With one held-out target per user: Recall@K = HitRate@K, NDCG@K = 1/log2(rank + 2) when
rank < K, MRR@K = 1/(rank + 1) when rank < K. Ranks are 0-based; ties are broken by item
index, so constant scorers (e.g. popularity with equal counts) are not flattered.
"""
from __future__ import annotations

import math

import torch

KS = (10, 20)


def target_ranks(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """scores: (B, N) with seen items already set to -inf; targets: (B,) column indices."""
    t = scores.gather(1, targets[:, None])
    greater = (scores > t).sum(1)
    cols = torch.arange(scores.shape[1], device=scores.device)[None, :]
    ties_before = ((scores == t) & (cols < targets[:, None])).sum(1)
    return greater + ties_before


class RankAccumulator:
    """Collects ranks batch by batch and reports averaged metrics."""

    def __init__(self, n_items: int | None = None, coverage_k: int = 10):
        self.ranks: list[torch.Tensor] = []
        self.n_items, self.coverage_k = n_items, coverage_k
        self.recommended = None

    def add(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        r = target_ranks(scores, targets)
        self.ranks.append(r.cpu())
        if self.n_items is not None:
            if self.recommended is None:
                self.recommended = torch.zeros(self.n_items, dtype=torch.bool, device=scores.device)
            self.recommended[scores.topk(self.coverage_k, dim=1).indices.flatten()] = True
        return r

    def result(self, mask: torch.Tensor | None = None) -> dict:
        r = torch.cat(self.ranks) if self.ranks else torch.zeros(0, dtype=torch.long)
        if mask is not None:
            r = r[mask]
        out = summarize(r)
        if self.recommended is not None and mask is None:
            out[f"coverage@{self.coverage_k}"] = round(self.recommended.float().mean().item(), 5)
        return out


def summarize(ranks: torch.Tensor) -> dict:
    n = int(ranks.numel())
    out: dict = {"users": n}
    if n == 0:
        return out
    r = ranks.double()
    for k in KS:
        hit = r < k
        out[f"recall@{k}"] = round(hit.double().mean().item(), 5)
        out[f"ndcg@{k}"] = round((hit.double() / torch.log2(r + 2)).mean().item(), 5)
    out["mrr@20"] = round(((r < 20).double() / (r + 1)).mean().item(), 5)
    out["median_rank"] = float(r.median().item())
    return out


def ndcg_single(rank: int, k: int) -> float:
    """Reference implementation used by the tests."""
    return 1.0 / math.log2(rank + 2) if rank < k else 0.0

import math

import torch

from src.metrics import RankAccumulator, ndcg_single, summarize, target_ranks


def test_ranks_break_ties_by_index():
    scores = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 3.0, 2.0, float("-inf")]])
    r = target_ranks(scores, torch.tensor([2, 2]))
    assert r.tolist() == [2, 1]                   # two equal items before it / one higher score


def test_masked_items_never_outrank():
    scores = torch.tensor([[5.0, float("-inf"), 1.0]])
    assert target_ranks(scores, torch.tensor([2])).item() == 1


def test_summary_values():
    out = summarize(torch.tensor([0, 1, 9, 10, 25]))
    assert math.isclose(out["recall@10"], 3 / 5, abs_tol=1e-5)
    assert math.isclose(out["recall@20"], 4 / 5, abs_tol=1e-5)
    exp = (ndcg_single(0, 10) + ndcg_single(1, 10) + ndcg_single(9, 10)) / 5
    assert math.isclose(out["ndcg@10"], exp, abs_tol=1e-5)
    assert math.isclose(out["mrr@20"], (1 + 1 / 2 + 1 / 10 + 1 / 11) / 5, abs_tol=1e-5)


def test_coverage():
    acc = RankAccumulator(n_items=30, coverage_k=10)
    s = torch.arange(30, dtype=torch.float32).repeat(2, 1)
    acc.add(s, torch.tensor([0, 29]))
    assert acc.result()["coverage@10"] == round(10 / 30, 5)

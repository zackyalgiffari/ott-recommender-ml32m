"""Render the results section of README.md and MODEL_CARD.md from the JSON results.

    python -m src.report --run-note "trained on a Colab NVIDIA L4 (bf16) in 74 minutes"

Reads results/metrics.json, results/train_history.json and (if present) results/ablations.json,
and replaces the text between <!-- RESULTS:START --> and <!-- RESULTS:END --> in both files,
so the published numbers are generated, never hand-copied.
"""
from __future__ import annotations

import argparse
import json
import re

from src.config import DOMAINS, HUB_REPO_ID, RESULTS_DIR, ROOT

LABEL = {"movielens": "MovieLens-32M", "amazon": "Amazon Movies & TV"}
COLS = [("recall@10", "Recall@10"), ("ndcg@10", "NDCG@10"), ("recall@20", "Recall@20"),
        ("ndcg@20", "NDCG@20"), ("mrr@20", "MRR@20"), ("coverage@10", "Coverage@10")]


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.2f}"


def row(name: str, m: dict, bold: bool = False) -> str:
    cells = [pct(m.get(k)) for k, _ in COLS]
    if bold:
        cells = [f"**{c}**" if c != "–" else c for c in cells]
    return f"| {name} | " + " | ".join(cells) + " |"


def render(run_note: str) -> str:
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    res = metrics["results"]
    hist = json.loads((RESULTS_DIR / "train_history.json").read_text())["history"]
    abl_path = RESULTS_DIR / "ablations.json"
    abl = json.loads(abl_path.read_text()) if abl_path.exists() else {}
    best = max(hist, key=lambda h: h["val_mean_ndcg@10"])

    out = [f"Model: [`{HUB_REPO_ID}`](https://huggingface.co/{HUB_REPO_ID}) — {run_note}; "
           f"best epoch {best['epoch']} of {len(hist)} by mean validation NDCG@10.", ""]
    out.append("Test set, **full-catalog ranking** within each domain (already-watched titles excluded); "
               "all values in %.")
    out.append("")
    for dom in DOMAINS:
        r = res[dom]
        n_users = r["hybrid"]["users"]
        out.append(f"**{LABEL[dom]}** — {n_users:,} test users")
        out.append("")
        out.append("| Model | " + " | ".join(c for _, c in COLS) + " |")
        out.append("|---|" + "---:|" * len(COLS))
        out.append(row("Hybrid — this model", r["hybrid"], bold=True))
        if "content_only" in r:
            out.append(row("Same model, content only (ID vectors off)", r["content_only"]))
        if "id_only" in abl and dom in abl["id_only"]:
            out.append(row("Ablation: ID-only SASRec", abl["id_only"][dom]))
        key = f"{dom}_only"
        if key in abl and dom in abl[key]:
            out.append(row(f"Ablation: hybrid trained on {LABEL[dom]} only", abl[key][dom]))
        out.append(row("Popularity baseline", r["popularity"]))
        out.append("")
    # cold-start slice
    out.append("**Cold titles** — test targets that never occur in any training sequence "
               "(only content can rank them):")
    out.append("")
    out.append("| Domain | Users | Hybrid R@20 | Content-only R@20 | Popularity R@20 |")
    out.append("|---|---:|---:|---:|---:|")
    for dom in DOMAINS:
        r = res[dom]
        c = r["hybrid"]["cold_targets"]
        out.append(f"| {LABEL[dom]} | {c['users']:,} | {pct(c.get('recall@20'))} | "
                   f"{pct(r.get('content_only', {}).get('cold_targets', {}).get('recall@20'))} | "
                   f"{pct(r['popularity']['cold_targets'].get('recall@20'))} |")
    out.append("")
    if (RESULTS_DIR / "results_ndcg.png").exists():
        out.append("![Test NDCG@10 by model variant](results/results_ndcg.png)")
        out.append("")
    out.append("<details><summary>Validation per epoch (20,000 sampled users per domain)</summary>")
    out.append("")
    out.append("| Epoch | Train loss | MovieLens NDCG@10 | Amazon NDCG@10 | Mean | Train time |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    for h in hist:
        v = h["val"]
        out.append(f"| {h['epoch']} | {h['loss']:.4f} | {pct(v['movielens']['ndcg@10'])} | "
                   f"{pct(v['amazon']['ndcg@10'])} | {pct(h['val_mean_ndcg@10'])} | {h['train_seconds'] // 60} min |")
    out.append("")
    out.append("</details>")
    return "\n".join(out)


def splice(path, body: str) -> None:
    text = path.read_text()
    new, n = re.subn(r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)",
                     lambda m: m.group(1) + body + m.group(2), text, flags=re.S)
    if n != 1:
        raise RuntimeError(f"results markers not found in {path}")
    path.write_text(new)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-note", required=True)
    args = p.parse_args()
    body = render(args.run_note)
    for f in ("README.md", "MODEL_CARD.md"):
        splice(ROOT / f, body)
    print(body)


if __name__ == "__main__":
    main()

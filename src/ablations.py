"""Merge the per-ablation evaluation files into results/ablations.json.

    python -m src.ablations            # reads results/ablations/<name>.json

Output: {name: {domain: <test metrics of that model>}}, e.g. ablations["id_only"]["amazon"]["ndcg@10"].
"""
from __future__ import annotations

import json

from src.config import RESULTS_DIR

DESCRIPTIONS = {
    "id_only": "SASRec with learned ID vectors only (no content tower), trained on both domains",
    "movielens_only": "hybrid model trained on MovieLens only",
    "amazon_only": "hybrid model trained on Amazon only",
    "id_dropout_0.05": "hybrid trained on both domains with ID-dropout 0.05 instead of 0.2 "
                       "(higher known-title accuracy, much weaker content-only / cold-start ranking)",
}


def main() -> None:
    out = {"_description": DESCRIPTIONS}
    for name in DESCRIPTIONS:
        f = RESULTS_DIR / "ablations" / f"{name}.json"
        if f.exists():
            res = json.loads(f.read_text())["results"]
            out[name] = {dom: r["hybrid"] for dom, r in res.items()}
            for dom, r in res.items():
                if "content_only" in r and name == "id_dropout_0.05":
                    out[name][dom]["content_only"] = r["content_only"]
    (RESULTS_DIR / "ablations.json").write_text(json.dumps(out, indent=2))
    print(f"[ablations] merged {[k for k in out if not k.startswith('_')]} -> {RESULTS_DIR / 'ablations.json'}")


if __name__ == "__main__":
    main()

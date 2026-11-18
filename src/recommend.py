"""Recommend what to watch next with the OTT hybrid recommender.

Command line (downloads the model from the Hub on first use):

    python recommend.py --domain movielens "The Matrix (1999)" "Inception (2010)" "Interstellar (2014)"
    python recommend.py --domain movielens "Toy Story (1995)" "Up (2009)" \
        --new-title "Paw Patrol: The Movie (2021). Genres: Animation, Children. Pups save Adventure City." \
        --new-title "Saw X (2023). Genres: Horror, Thriller. John Kramer returns."

Python:

    from recommend import OTTRecommender
    rec = OTTRecommender("zackyalgiffari/ott-recommender-ml32m")
    rec.recommend(["The Matrix (1999)", "Inception (2010)"], domain="movielens", k=10)
    rec.rank_new_titles(["The Matrix (1999)"], ["<title>. Genres: ... <plot>"], domain="movielens")

History entries may be catalog titles ("Title (Year)"), item ids such as "ml:2571" or
"az:B00005JNQV", or integer item indices. ``rank_new_titles`` scores titles that are not in
the catalog from their text alone (cold start), using the same encoder the model was trained with.
"""
from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

try:  # inside the Hugging Face model repository
    from modeling import HybridSASRec
except ImportError:  # inside the GitHub repository
    from src.model import HybridSASRec

REPO_ID = "zackyalgiffari/ott-recommender-ml32m"
PREFIX = {"movielens": "ml", "amazon": "az"}


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", str(s).casefold()).strip()


class OTTRecommender:
    def __init__(self, model: str | Path = REPO_ID, device: str = "cpu", revision: str | None = None):
        path = Path(model)
        if not path.exists():
            path = Path(snapshot_download(str(model), revision=revision))
        self.path, self.device = path, torch.device(device)
        self.model = HybridSASRec.from_pretrained(str(path))
        if self.model.use_content:
            self.model.set_content(load_file(str(path / "item_content.safetensors"))["content"])
        self.model.to(self.device).eval()
        cat = pd.read_parquet(path / "catalog.parquet")
        self.catalog = cat.set_index("item_idx", drop=False)
        self._by_id, self._by_title_year, self._by_title = {}, {}, {}
        cat = cat.sort_values("popularity", ascending=False)  # ambiguous titles -> most watched
        for dom, g in cat.groupby("domain"):
            self._by_id[dom] = dict(zip(PREFIX[dom] + ":" + g["source_id"], g["item_idx"]))
            ty = (g["title"].map(_norm) + " " + g["year"].astype("Int64").astype(str).str.replace("<NA>", "")).str.strip()
            self._by_title_year[dom] = dict(zip(ty[::-1], g["item_idx"][::-1]))  # keep most watched
            self._by_title[dom] = dict(zip(g["title"].map(_norm)[::-1], g["item_idx"][::-1]))
        self._encoder = None

    # ------------------------------------------------------------------ lookup
    def resolve(self, entry: str | int, domain: str) -> int:
        """Map a title / item id / index to the catalog item_idx of ``domain``."""
        lo, hi = self.model.domain_ranges[self.model.domains.index(domain)]
        if isinstance(entry, int) or (isinstance(entry, str) and entry.isdigit()):
            idx = int(entry)
            if not lo <= idx < hi:
                raise KeyError(f"item index {idx} is not a {domain} item")
            return idx
        if entry in self._by_id[domain]:
            return int(self._by_id[domain][entry])
        key = _norm(entry)
        for table in (self._by_title_year[domain], self._by_title[domain]):
            if key in table:
                return int(table[key])
        close = difflib.get_close_matches(key, self._by_title_year[domain].keys(), n=1, cutoff=0.85)
        if close:
            return int(self._by_title_year[domain][close[0]])
        hint = difflib.get_close_matches(key, self._by_title_year[domain].keys(), n=3, cutoff=0.5)
        raise KeyError(f"{entry!r} not found in the {domain} catalog; close: {hint}")

    def _user_state(self, history: list, domain: str) -> tuple[torch.Tensor, list[int]]:
        idx = [self.resolve(h, domain) for h in history]
        if not idx:
            raise ValueError("history must contain at least one title")
        seq = torch.tensor([idx[-self.model.max_len:]], device=self.device)
        dom = torch.tensor([self.model.domains.index(domain)], device=self.device)
        with torch.no_grad():
            return self.model.user_state(seq, dom)[0], idx

    # ------------------------------------------------------------------ recommend
    @torch.no_grad()
    def recommend(self, history: list, domain: str = "movielens", k: int = 10,
                  exclude_seen: bool = True) -> pd.DataFrame:
        """Top-k catalog titles to watch next after ``history`` (oldest first)."""
        h, idx = self._user_state(history, domain)
        d = self.model.domains.index(domain)
        lo, _ = self.model.domain_ranges[d]
        scores = self.model.domain_items(d) @ h
        if exclude_seen:
            scores[torch.tensor(idx, device=self.device) - lo] = float("-inf")
        top = scores.topk(k)
        rows = self.catalog.loc[(top.indices + lo).tolist()].copy()
        rows["score"] = [round(x, 4) for x in top.values.tolist()]
        rows["year"] = rows["year"].astype("Int64")
        cols = ["item_idx", "source_id", "title", "year", "genres", "fmt", "score"]
        return rows[cols].reset_index(drop=True)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        from transformers import AutoModel, AutoTokenizer
        if self._encoder is None:
            self._encoder = (AutoTokenizer.from_pretrained(self.model.encoder),
                             AutoModel.from_pretrained(self.model.encoder).to(self.device).eval())
        tok, enc = self._encoder
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        cls = enc(**batch).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1)

    @torch.no_grad()
    def rank_new_titles(self, history: list, texts: list[str], domain: str = "movielens") -> pd.DataFrame:
        """Score titles that are not in the catalog from their text alone (cold start).

        Describe each title like the training data: "Title (Year). Genres: A, B. Plot ...".
        """
        if not self.model.use_content:
            raise RuntimeError("this model has no content tower")
        h, _ = self._user_state(history, domain)
        vec = self.model.content_vectors(self.embed_texts(texts))
        return (pd.DataFrame({"text": texts, "score": (vec @ h).tolist()})
                .sort_values("score", ascending=False).reset_index(drop=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("history", nargs="+", help="watched titles, oldest first")
    p.add_argument("--model", default=REPO_ID, help="Hub repo id or local folder")
    p.add_argument("--domain", default="movielens", choices=list(PREFIX))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--new-title", action="append", default=[], help="text of a title not in the catalog")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    rec = OTTRecommender(args.model, device=args.device)
    names = [rec.catalog.loc[rec.resolve(h, args.domain), "title"] for h in args.history]
    print(f"History ({args.domain}): " + " -> ".join(names))
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(rec.recommend(args.history, args.domain, args.k).to_string(index=False))
        if args.new_title:
            print("\nNew titles ranked from text only (cold start):")
            print(rec.rank_new_titles(args.history, args.new_title, args.domain).to_string(index=False))


if __name__ == "__main__":
    main()

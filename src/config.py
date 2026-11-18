"""Shared paths, pinned data sources and constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Large data stays on a fast local disk: ~/.cache locally, /content/data on Colab.
DATA_DIR = Path(os.environ.get("OTT_DATA_DIR", Path.home() / ".cache" / "ott-recommender" / "data"))
RESULTS_DIR = ROOT / "results"

HUB_REPO_ID = "zackyalgiffari/ott-recommender-ml32m"
GITHUB_URL = "https://github.com/zackyalgiffari/ott-recommender-ml32m"
ENCODER = "BAAI/bge-base-en-v1.5"

# Domain ids are positions in this tuple. Items of a domain occupy one contiguous
# index range in the unified catalog (index 0 is padding).
DOMAINS = ("movielens", "amazon")
ML, AZ = 0, 1
ITEM_PREFIX = {ML: "ml", AZ: "az"}

# Every source is pinned to a commit, so a local run and a Colab run read the same bytes.
SOURCES = {
    "ml32m": {
        "repo_id": "hazemessam/ml-32m",
        "revision": "42dc8193d723e5de13316f4b663ee5024d87f573",
        "files": ["ratings.csv", "movies.csv", "links.csv", "checksums.txt", "README.txt"],
    },
    "tmdb": {
        "repo_id": "ada-datadruids/full_tmdb_movies_dataset",
        "revision": "13ba9f3a87c60eaa07b235e1cfd2e16cc1c9105c",
        "files": ["TMDB_movie_dataset_v11.csv"],
    },
    "amazon": {
        "repo_id": "McAuley-Lab/Amazon-Reviews-2023",
        "revision": "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e",
        "files": [
            "benchmark/5core/rating_only/Movies_and_TV.csv",
            "raw/meta_categories/meta_Movies_and_TV.jsonl",
        ],
    },
}

# Published in the ML-32M README; prepare.py asserts them.
ML32M_OFFICIAL = {"ratings": 32_000_204, "movies": 87_585, "users": 200_948}


def processed_dir(sample_users: int | None = None) -> Path:
    """Where prepare.py writes; smoke runs get their own directory."""
    return DATA_DIR / ("processed" if not sample_users else f"processed-sample{sample_users}")

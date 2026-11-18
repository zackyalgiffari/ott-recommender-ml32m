import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def toy_processed(tmp_path: Path) -> Path:
    """Two domains: items 1..6 (movielens), 7..10 (amazon); four users."""
    users = [
        (0, [1, 2, 3, 4, 5, 6]),      # movielens, 6 events
        (0, [3, 1, 5]),               # movielens, 3 events (minimum)
        (1, [7, 8, 9, 10]),           # amazon
        (1, [10, 9, 7]),
    ]
    flat = np.concatenate([np.array(s, dtype=np.int32) for _, s in users])
    offsets = np.concatenate([[0], np.cumsum([len(s) for _, s in users])]).astype(np.int64)
    domain = np.array([d for d, _ in users], dtype=np.int8)
    np.savez(tmp_path / "sequences.npz", items=flat, offsets=offsets, domain=domain)
    meta = {"n_items": 10, "domains": ["movielens", "amazon"], "domain_ranges": [[1, 7], [7, 11]],
            "split_sha256": "toy", "sample_users": None}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    titles = ["Toy Story", "Heat", "Up", "Alien", "Brazil", "Fargo", "Ref", "Saw", "Big", "Jaws"]
    pl.DataFrame({
        "item_idx": pl.Series(range(1, 11), dtype=pl.Int32),
        "domain": pl.Series([0] * 6 + [1] * 4, dtype=pl.Int8),
        "source_id": [str(i) for i in range(1, 7)] + [f"B0{i}" for i in range(7, 11)],
        "title": titles,
        "year": pl.Series([1995, 1995, 2009, 1979, 1985, 1996, 1994, 2004, 1988, 1975], dtype=pl.Int32),
        "genres": ["g"] * 10, "fmt": ["Movie"] * 6 + ["DVD"] * 4, "has_title": [True] * 10,
        "tmdb_id": pl.Series([None] * 10, dtype=pl.Int64), "language": pl.Series([None] * 10, dtype=pl.Utf8),
        "text": [f"{t}." for t in titles],
    }).write_parquet(tmp_path / "items.parquet")
    g = torch.Generator().manual_seed(0)
    content = torch.nn.functional.normalize(torch.randn(11, 16, generator=g), dim=-1)
    content[0] = 0
    save_file({"content": content.half()}, str(tmp_path / "item_content.safetensors"),
              metadata={"encoder": "toy"})
    return tmp_path

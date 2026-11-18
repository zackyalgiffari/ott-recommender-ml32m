from src.model import HybridSASRec
from src.package import build
from src.recommend import OTTRecommender


def test_package_and_recommend(toy_processed, tmp_path):
    m = HybridSASRec(n_items=10, content_dim=16, d_model=32, n_layers=1, n_heads=2, max_len=8,
                     domain_ranges=[[1, 7], [7, 11]], encoder="toy")
    m.save_pretrained(tmp_path / "best")
    out = build(tmp_path / "best", toy_processed, tmp_path / "hub")
    for f in ("config.json", "model.safetensors", "item_content.safetensors", "catalog.parquet",
              "modeling.py", "recommend.py", "requirements.txt"):
        assert (out / f).exists(), f
    rec = OTTRecommender(out)
    assert rec.resolve("Heat (1995)", "movielens") == 2
    assert rec.resolve("ml:4", "movielens") == 4
    assert rec.resolve("alien", "movielens") == 4
    assert rec.resolve("az:B09", "amazon") == 9
    top = rec.recommend(["Toy Story (1995)", "Heat"], "movielens", k=3)
    assert len(top) == 3 and not set(top["item_idx"]) & {1, 2}
    assert set(top["item_idx"]) <= set(range(1, 7))
    top_az = rec.recommend(["Saw"], "amazon", k=2)
    assert set(top_az["item_idx"]) <= set(range(7, 11))

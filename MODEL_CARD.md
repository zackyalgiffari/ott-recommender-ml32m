---
license: cc-by-nc-4.0
library_name: pytorch
language:
- en
tags:
- recommender-system
- sequential-recommendation
- sasrec
- ott
- movies
- tv-shows
- cold-start
- pytorch_model_hub_mixin
datasets:
- hazemessam/ml-32m
- ada-datadruids/full_tmdb_movies_dataset
- McAuley-Lab/Amazon-Reviews-2023
---

# OTT next-to-watch recommender (MovieLens-32M + Amazon Movies & TV)

Given what a viewer watched, in order, this model ranks **what they will watch next** over a
catalog of **285,528 movies and TV titles** from two sources:

- 87,585 MovieLens movies, with real TMDB metadata.
- 197,943 Amazon Movies & TV titles, about half of them Prime Video, including series such as
  *Justified*, *Vikings* and *Downton Abbey*.

It is a SASRec-style causal transformer. Each title's vector combines a learned ID embedding
with a projected [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5)
embedding of its metadata text. Because the ID part is randomly dropped during training, the
model can also rank **new titles described only by text** (cold start).

Code, data pipeline and training logs: <https://github.com/zackyalgiffari/ott-recommender-ml32m>

<!-- RESULTS:START -->
_Results are added after the full training run._
<!-- RESULTS:END -->

## How to use

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("zackyalgiffari/ott-recommender-ml32m")
sys.path.insert(0, path)
from recommend import OTTRecommender

rec = OTTRecommender(path)            # device="cuda" if available
rec.recommend(["The Matrix (1999)", "Toy Story (1995)", "Inception (2010)"], domain="movielens", k=10)
rec.recommend(["Justified", "Vikings", "The Tomorrow War"], domain="amazon", k=10)

# Cold start: score titles that are not in the catalog, from their text alone
rec.rank_new_titles(
    ["The Matrix (1999)", "Inception (2010)"],
    ["Dune: Part Two (2024). Genres: Sci-Fi, Adventure. Paul Atreides unites with the Fremen ...",
     "The Holiday Romance (2024). Genres: Comedy, Romance. Two strangers swap homes ..."],
    domain="movielens",
)
```

Command line, from a clone of the repo or a downloaded snapshot:

```bash
pip install -r requirements.txt
python recommend.py --domain movielens "The Matrix (1999)" "Toy Story (1995)" "Inception (2010)"
```

History entries can be catalog titles (`"Title (Year)"`; fuzzy matching is applied), item ids
such as `ml:2571` / `az:B00005JNQV`, or integer item indices. Recommendations are ranked within
the chosen domain's catalog, and already-watched titles are excluded.

| File | Content |
|---|---|
| `config.json`, `model.safetensors` | model hyper-parameters and weights (`PyTorchModelHubMixin`) |
| `item_content.safetensors` | frozen bge-base embeddings of every catalog title (float16) |
| `catalog.parquet` | `item_idx`, domain, source id, title, year, genres, format, training popularity |
| `modeling.py`, `recommend.py` | self-contained model class and inference helper |
| `results/` | data report, test metrics, training history, ablations |

## Data

| Source | Use |
|---|---|
| [MovieLens-32M](https://huggingface.co/datasets/hazemessam/ml-32m) | 32,000,204 ratings by 200,948 users (1995–2023), used as watch events |
| [TMDB movie dump](https://huggingface.co/datasets/ada-datadruids/full_tmdb_movies_dataset) | overview, tagline, keywords, languages and runtime for 86,337 MovieLens movies |
| [Amazon Reviews 2023, Movies & TV](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | 5-core: 7,441,129 reviews by 657,203 users (1997–2023), plus item metadata |

Data-quality steps (all recorded in `results/data_report.json`):

- **MovieLens integrity.** The MD5 of every CSV matches the official `checksums.txt`, and the
  rating, movie and user counts equal the published ones.
- **Validated TMDB join.** TMDB metadata is attached only when MovieLens's `imdbId` agrees with
  TMDB's `imdb_id`. This rejected 148 wrong `tmdbId` links. In total 98.6% of movies got
  validated metadata.
- **No future information.** All-time aggregates (Amazon `average_rating` / `rating_number`,
  MovieLens tags) are not used as features.
- **Amazon titles.** 82,840 Prime Video items have no title or description in the source
  metadata. They are kept, because they are in many users' histories, and are represented by
  director, cast, content advisory and language. In the catalog they appear as
  `Untitled Prime Video item (dir. …)`.

## Training

- **Split.** Per-user leave-last-out by timestamp. Test = last event, validation =
  second-to-last, training = the rest; ties are broken by item index. The split's SHA-256 is
  reported, so runs are verifiable.
- **Architecture.** 2 pre-LN transformer blocks, d = 256, 4 heads, dropout 0.2. Domain and
  position embeddings. Histories are up to 200 (MovieLens) or 50 (Amazon) titles long.
- **Item vectors.** `W · bge(text) + E_id`, with ID-dropout p = 0.2.
- **Objective.** Next-title sampled softmax with 8,192 uniform negatives from the same domain.
  Every batch holds a single domain.
- **Optimisation.** AdamW (lr 1e-3, warm-up, cosine decay), bf16, 25,600 tokens per batch.
  Early stopping on mean validation NDCG@10 (patience 3).
- **Hardware.** One NVIDIA L4 on Google Colab, launched with the `colab` CLI.

## Evaluation

For every user, the held-out title is ranked against the **entire catalog of its domain**
(87,585 or 197,943 titles), excluding titles the user already watched. No sampled candidate
lists are used. Reported: Recall@10/20, NDCG@10/20, MRR@20, catalog coverage@10, and the
same metrics on *cold* targets that never appear in training.

## Limitations

- **Ratings and reviews stand in for views.** MovieLens ratings and Amazon reviews are proxies
  for viewing. Many Amazon items are DVD/Blu-ray products, not streams.
- **Popularity bias.** Both sources over-represent popular and English-language titles. Plan
  for re-ranking if diversity matters.
- **Leave-last-out timing.** Every user's last event is the test target. Different users' test
  events come from different years, so this split is not a strict global-time split.
- **No cross-domain merging.** A movie that exists in both catalogs is two separate items.
- **English text encoder.** Metadata for non-English titles is embedded with an English
  model, so content similarity is weaker for those titles.

## License and attribution

The model is released under **CC BY-NC 4.0** because it is derived from MovieLens (research,
non-commercial), Amazon Reviews 2023 (research use) and TMDB metadata. This product uses
TMDB data but is not endorsed or certified by TMDB.

- F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and
  Context.* ACM TiiS 5(4).
- Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, Julian McAuley. 2024. *Bridging
  Language and Items for Retrieval and Recommendation.* arXiv:2403.03952.
- Wang-Cheng Kang and Julian McAuley. 2018. *Self-Attentive Sequential Recommendation.* ICDM.
- Shitao Xiao et al. 2023. *C-Pack: Packaged Resources to Advance General Chinese Embedding*
  (BGE). arXiv:2309.07597.

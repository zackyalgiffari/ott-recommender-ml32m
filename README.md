# ott-recommender-ml32m

A **next-to-watch recommender for OTT catalogs**, trained on two public sources from the
Hugging Face Hub: **MovieLens-32M** (with real TMDB metadata) and **Amazon Reviews 2023 —
Movies & TV** (movies and TV series, about half of them Prime Video titles). Training runs on
a Google Colab **L4** GPU through the [`colab` CLI](https://pypi.org/project/google-colab-cli/).

The model is a SASRec-style causal transformer over a user's watch history. Every title is
represented by a learned ID vector **plus** a projected text embedding of its metadata from
[`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5). Because the ID part
is randomly dropped during training, the content path also works alone, so the model can
recommend brand-new titles from their description (cold start).

Model on the Hub: [`zackyalgiffari/ott-recommender-ml32m`](https://huggingface.co/zackyalgiffari/ott-recommender-ml32m)

## Results

<!-- RESULTS:START -->
_Results are added after the full training run._
<!-- RESULTS:END -->

## Data

| Source (HF Hub) | What it contributes |
|---|---|
| [`hazemessam/ml-32m`](https://huggingface.co/datasets/hazemessam/ml-32m) | 32,000,204 ratings by 200,948 users on 87,585 movies (1995–2023), used as watch events |
| [`ada-datadruids/full_tmdb_movies_dataset`](https://huggingface.co/datasets/ada-datadruids/full_tmdb_movies_dataset) | Plot overview, keywords, languages and runtime for MovieLens titles |
| [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | Movies & TV 5-core: 7,441,129 reviews by 657,203 users on 197,943 titles, plus item metadata |

Every source is pinned to a commit (`src/config.py`). `src/prepare.py` checks data quality
and records every check in [`results/data_report.json`](results/data_report.json):

- **MovieLens integrity.** The MD5 of every CSV must match the official `checksums.txt`, and
  the rating, movie and user counts must equal the numbers published in the release README.
- **Validated TMDB join.** A TMDB record is attached only when MovieLens's `imdbId` matches
  TMDB's `imdb_id`. This catches 148 wrong `tmdbId` links in MovieLens itself. Records without
  an IMDb id need an identical normalised title and a year within ±1.
- **No future information in features.** Amazon's `average_rating` / `rating_number` and
  MovieLens tags are aggregates over all time, including the held-out events, so they are
  **not** used.
- **Deterministic split.** Per-user leave-last-out by timestamp, with ties broken by item index.
  The SHA-256 of the split arrays is reported, so a Colab run can be checked against a local one.

## Method

```
item vector   v_i = W · bge(text_i) + E_id[i]          (ID part dropped with p = 0.2 while training)
user state    h_u = Transformer(v_{i1} + pos_1 + dom, ..., v_{in} + pos_n + dom)[last]
score         s(u, i) = <h_u, v_i>                     (ranked within the user's domain catalog)
```

- 2 pre-LN transformer blocks, d = 256, 4 heads, dropout 0.2. History windows are up to 200
  (MovieLens) or 50 (Amazon) titles long.
- Sampled softmax with 8,192 uniform negatives from the same domain. Each batch holds a single
  domain, so the model never learns the trivial "which catalog is this" signal.
- Evaluation ranks the held-out title against the **full catalog** of its domain, with
  already-watched titles removed. No sampled-candidate metrics are used.

## Reproduce

```bash
uv venv ~/.venvs/ott-prediction-model --python 3.12
uv pip install --python ~/.venvs/ott-prediction-model/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ~/.venvs/ott-prediction-model/bin/python -r requirements.txt pytest
python -m pytest -q                                       # unit tests

python -m src.prepare                                     # data + data_report.json (CPU, ~1 min)
scripts/colab_train.sh smoke                              # end-to-end check on a Colab L4
scripts/colab_train.sh full --push-to-hub <user>/<repo>   # full training run
scripts/colab_train.sh ablations                          # ID-only / single-domain models
```

| Stage | Module |
|---|---|
| download + MD5 verification | `src/download.py` |
| cleaning, TMDB join, item texts, split | `src/prepare.py` |
| text embeddings (bge-base) | `src/embed_items.py` |
| model | `src/model.py` |
| training | `src/train.py` |
| full-catalog evaluation | `src/evaluate.py`, `src/metrics.py` |
| Hub packaging | `src/package.py` |
| inference / demo | `src/recommend.py` |

## License

Code: MIT. The trained model and its catalog derive from MovieLens (non-commercial research
use), Amazon Reviews 2023 (research use) and TMDB metadata, so the model is released under
**CC BY-NC 4.0**.

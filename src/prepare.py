"""Build the unified two-domain catalog, the user watch sequences and the leave-last-out split.

    python -m src.prepare                      # full data -> DATA_DIR/processed, results/data_report.json
    python -m src.prepare --sample-users 2000  # deterministic subset per domain, for smoke tests

Outputs (in ``processed_dir``):
    items.parquet   one row per catalog item; item_idx 1..N, ML items first, then Amazon
    sequences.npz   items (int32, flat, time-ordered per user), offsets (int64), domain (int8)
    meta.json       catalog size, contiguous index range of each domain, split hash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from src.config import AZ, DOMAINS, ML, ML32M_OFFICIAL, RESULTS_DIR, processed_dir
from src.download import fetch_all

MIN_EVENTS = 3  # val target + test target + at least one history item
MAX_OVERVIEW = 1000
MAX_KEYWORDS = 20

# ---------------------------------------------------------------------------- MovieLens titles
_YEAR = re.compile(r"\s*\((\d{4})(?:[-–]\d{0,4})?\)\s*$")
_ARTICLE = re.compile(
    r"^(?P<body>.+), (?P<art>The|A|An|Les|La|Le|Il|El|Lo|Gli|Die|Der|Das|Los|Las|Un|Une|De|Het|Een|Det|Den)$"
)


def split_ml_title(raw: str) -> tuple[str, int | None]:
    """'City of Lost Children, The (Cité des enfants perdus, La) (1995)'
    -> ('The City of Lost Children (Cité des enfants perdus, La)', 1995)"""
    raw = raw.strip()
    m = _YEAR.search(raw)
    year = int(m.group(1)) if m else None
    title = raw[: m.start()] if m else raw
    main, alt = title, ""
    if title.endswith(")") and " (" in title:  # alternative / original-language title
        cut = title.index(" (")
        main, alt = title[:cut], title[cut:]
    a = _ARTICLE.match(main)
    if a:
        main = f"{a['art']} {a['body']}"
    return (main + alt).strip(), year


def _norm(s: str | None) -> str:
    return re.sub(r"[^0-9a-z]+", "", (s or "").casefold())


# ---------------------------------------------------------------------------- MovieLens + TMDB
def build_movielens(paths: dict, tmdb_path: Path, report: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (items, events). events: user (str), source_id (str), ts (int64 seconds)."""
    ratings = pl.read_csv(
        paths["ratings.csv"],
        schema_overrides={"userId": pl.Int64, "movieId": pl.Int64, "rating": pl.Float32, "timestamp": pl.Int64},
    )
    movies = pl.read_csv(paths["movies.csv"], schema_overrides={"movieId": pl.Int64})
    links = pl.read_csv(paths["links.csv"], schema_overrides={"movieId": pl.Int64, "imdbId": pl.Utf8, "tmdbId": pl.Int64})

    counts = {"ratings": ratings.height, "movies": movies.height, "users": ratings["userId"].n_unique()}
    if counts != ML32M_OFFICIAL:
        raise RuntimeError(f"MovieLens counts {counts} differ from the official {ML32M_OFFICIAL}")
    rep = report["movielens"] = {
        "official_counts_ok": True,
        **counts,
        "movies_with_ratings": ratings["movieId"].n_unique(),
        "rating_time_range_utc": [
            str(pl.from_epoch(pl.Series([ratings["timestamp"].min()]), time_unit="s")[0]),
            str(pl.from_epoch(pl.Series([ratings["timestamp"].max()]), time_unit="s")[0]),
        ],
    }

    parsed = [split_ml_title(t) for t in movies["title"].to_list()]
    movies = movies.with_columns(
        pl.Series("clean_title", [p[0] for p in parsed]),
        pl.Series("year", [p[1] for p in parsed], dtype=pl.Int32),
        pl.when(pl.col("genres") == "(no genres listed)").then(pl.lit("")).otherwise(
            pl.col("genres").str.replace_all(r"\|", ", ")).alias("genre_str"),
    ).join(links, on="movieId", how="left")

    # TMDB: every column read as text (the dump has free text with embedded newlines).
    tmdb = (
        pl.read_csv(tmdb_path, infer_schema_length=0)
        .with_columns(pl.col("id").cast(pl.Int64, strict=False))
        .filter(pl.col("id").is_not_null())
        .unique("id", keep="first", maintain_order=True)
        .select(
            pl.col("id").alias("tmdbId"),
            pl.col("title").alias("tmdb_title"),
            pl.col("release_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("tmdb_year"),
            pl.col("imdb_id").alias("tmdb_imdb"),
            "original_language", "spoken_languages", "runtime", "tagline", "overview", "keywords",
        )
    )
    j = movies.join(tmdb, on="tmdbId", how="left").with_columns(("tt" + pl.col("imdbId")).alias("ml_imdb"))
    found = pl.col("tmdb_title").is_not_null()
    tmdb_has_imdb = pl.col("tmdb_imdb").is_not_null() & (pl.col("tmdb_imdb") != "")
    imdb_match = found & tmdb_has_imdb & (pl.col("tmdb_imdb") == pl.col("ml_imdb"))
    imdb_mismatch = found & tmdb_has_imdb & (pl.col("tmdb_imdb") != pl.col("ml_imdb"))
    j = j.with_columns(
        imdb_match.alias("imdb_match"),
        imdb_mismatch.alias("imdb_mismatch"),
        (found & ~tmdb_has_imdb).alias("imdb_missing"),
    )
    # Without an IMDb id on the TMDB side, accept only an exact (normalised) title and a year within 1.
    title_ok = pl.Series(
        [_norm(a) == _norm(b) for a, b in zip(j["clean_title"].to_list(), j["tmdb_title"].to_list())]
    )
    year_close = (pl.col("year") - pl.col("tmdb_year")).abs() <= 1
    j = j.with_columns(title_ok.alias("title_ok")).with_columns(
        (pl.col("imdb_match") | (pl.col("imdb_missing") & pl.col("title_ok") & year_close.fill_null(False)))
        .alias("tmdb_ok")
    )
    year_gap = (pl.col("year") - pl.col("tmdb_year")).abs()
    acc = j.filter("tmdb_ok")
    rep["tmdb_join"] = {
        "rule": "accept when MovieLens imdbId == TMDB imdb_id; if TMDB has no imdb_id, require "
                "an identical normalised title and |year gap| <= 1",
        "tmdb_dump_rows_unique_ids": tmdb.height,
        "ml_movies_with_tmdbId": int(j["tmdbId"].is_not_null().sum()),
        "found_in_dump": int(j.select(found.sum()).item()),
        "imdb_match": int(j["imdb_match"].sum()),
        "imdb_mismatch_rejected": int(j["imdb_mismatch"].sum()),
        "imdb_missing": int(j["imdb_missing"].sum()),
        "imdb_missing_accepted_by_title_year": int(j.filter(pl.col("imdb_missing") & pl.col("tmdb_ok")).height),
        "accepted": acc.height,
        "accepted_share": round(acc.height / j.height, 4),
        "accepted_year_gap_ge2": int(acc.select((year_gap >= 2).fill_null(False).sum()).item()),
        "accepted_with_overview": int(acc.select((pl.col("overview").str.len_chars() > 20).fill_null(False).sum()).item()),
        "accepted_with_keywords": int(acc.select((pl.col("keywords").str.len_chars() > 0).fill_null(False).sum()).item()),
    }
    tm_cols = ["original_language", "spoken_languages", "runtime", "tagline", "overview", "keywords"]
    j = j.with_columns(
        [pl.when(pl.col("tmdb_ok")).then(pl.col(c)).otherwise(None).alias(c) for c in tm_cols]
        + [pl.when(pl.col("tmdb_ok")).then(pl.col("tmdbId")).otherwise(None).alias("tmdb_id"),
           pl.coalesce(pl.col("year"), pl.when(pl.col("tmdb_ok")).then(pl.col("tmdb_year"))).alias("year")]
    )

    rows = j.select("movieId", "clean_title", "year", "genre_str", "tmdb_id", *tm_cols).to_dicts()
    texts = [ml_text(r) for r in rows]
    items = pl.DataFrame({
        "domain": pl.Series([ML] * len(rows), dtype=pl.Int8),
        "source_id": [str(r["movieId"]) for r in rows],
        "title": [r["clean_title"] for r in rows],
        "year": pl.Series([r["year"] for r in rows], dtype=pl.Int32),
        "genres": [r["genre_str"] for r in rows],
        "fmt": ["Movie"] * len(rows),
        "has_title": [True] * len(rows),
        "tmdb_id": pl.Series([r["tmdb_id"] for r in rows], dtype=pl.Int64),
        "language": [r["original_language"] for r in rows],
        "text": texts,
    })
    rep["genres"] = dict(
        movies["genres"].str.split("|").explode().value_counts().sort("count", descending=True).iter_rows()
    )
    rep["original_language_top15"] = dict(
        items.filter(pl.col("language").is_not_null())["language"].value_counts()
        .sort("count", descending=True).head(15).iter_rows()
    )
    rep["distinct_original_languages"] = items["language"].drop_nulls().n_unique()
    rep["release_decades"] = _decades(items["year"])

    events = ratings.select(
        pl.col("userId").cast(pl.Utf8).alias("user"),
        pl.col("movieId").cast(pl.Utf8).alias("source_id"),
        pl.col("timestamp").alias("ts"),
    )
    return items, events


def _clip(s: str | None, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."


def ml_text(r: dict) -> str:
    parts = [f"{r['clean_title']} ({r['year']})." if r["year"] else f"{r['clean_title']}."]
    if r["genre_str"]:
        parts.append(f"Genres: {r['genre_str']}.")
    if r.get("spoken_languages"):
        parts.append(f"Language: {_clip(r['spoken_languages'], 80)}.")
    try:
        rt = int(float(r.get("runtime") or 0))
    except ValueError:
        rt = 0
    if rt > 0:
        parts.append(f"Runtime: {rt} min.")
    if r.get("tagline"):
        parts.append(_clip(r["tagline"], 200))
    if r.get("overview"):
        parts.append(_clip(r["overview"], MAX_OVERVIEW))
    if r.get("keywords"):
        kws = [k.strip() for k in r["keywords"].split(",") if k.strip()][:MAX_KEYWORDS]
        parts.append(f"Keywords: {', '.join(kws)}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------- Amazon Movies & TV
_GENERIC_CATS = {
    "Movies & TV", "Featured Categories", "Genre for Featured Categories", "Studio Specials", "All Titles",
    "Blu-ray", "DVD", "Movies", "TV", "Boxed Sets", "Independently Distributed", "Prime Video", "4K",
    "Widescreen", "Fully Loaded DVDs", "Special Editions", "Today's Deals in Movies & TV",
}
_STUDIO = re.compile(r"Home Entertainment|Home Video|Pictures|Studios?|Titles|Entertainment|Video$|Films?$")


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() and v.strip().lower() != "none" else []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip() and str(x).strip().lower() != "none"]
    return [str(v)]


def _az_year(det: dict) -> int | None:
    for key in ("Release date", "Date First Available"):
        for v in _as_list(det.get(key)):
            m = re.search(r"\b(18[89]\d|19\d\d|20[0-2]\d)\b", v)
            if m:
                return int(m.group(1))
    return None


def az_item(d: dict) -> dict:
    det = d.get("details") or {}
    title = (d.get("title") or "").strip()
    if title.lower() == "none":
        title = ""
    main_cat = d.get("main_category") or ""
    cats = _as_list(d.get("categories"))
    fmt_raw = " ".join(_as_list(det.get("Format")) + _as_list(det.get("Media Format")) + cats).lower()
    if main_cat == "Prime Video":
        fmt = "Prime Video"
    elif "blu-ray" in fmt_raw:
        fmt = "Blu-ray"
    elif "dvd" in fmt_raw:
        fmt = "DVD"
    elif "vhs" in fmt_raw:
        fmt = "VHS"
    else:
        fmt = "Disc (format unknown)"
    genres = _as_list(det.get("Genre"))
    genres += [c for c in cats if c not in _GENERIC_CATS and not _STUDIO.search(c)]
    genres = list(dict.fromkeys(g.strip() for g in genres if g.strip()))[:6]
    directors = list(dict.fromkeys(_as_list(det.get("Directors")) + _as_list(det.get("Director"))))[:3]
    cast = list(dict.fromkeys(_as_list(det.get("Starring")) + _as_list(det.get("Actors"))))[:5]
    # A year written in the title is the film's year; "Release date" of a disc is the disc release.
    m = _YEAR.search(title)
    year = int(m.group(1)) if m else _az_year(det)
    desc = _clip(" ".join(_as_list(d.get("description"))), MAX_OVERVIEW)
    if title:
        display = title
    elif directors or cast:
        who = ([f"dir. {directors[0]}"] if directors else []) + ([f"with {', '.join(cast[:2])}"] if cast else [])
        display = f"Untitled {fmt} item ({'; '.join(who)})"
    else:
        display = f"Untitled {fmt} item {d['parent_asin']}"
    parts = [f"{title}." if title else f"{fmt} title."]
    if year and not m:
        parts.append(f"Released: {year}.")
    parts.append(f"Format: {fmt}.")
    if genres:
        parts.append(f"Genres: {', '.join(genres)}.")
    if directors:
        parts.append(f"Directed by {', '.join(directors)}.")
    if cast:
        parts.append(f"Starring {', '.join(cast)}.")
    adv = _as_list(det.get("Content advisory"))
    if adv:
        parts.append(f"Content advisory: {_clip(', '.join(adv), 150)}.")
    lang = _as_list(det.get("Audio languages")) or _as_list(det.get("Language"))
    lang = [re.sub(r"\s*\([^)]*\)", "", x).strip() for x in ", ".join(lang).split(",")]
    lang = list(dict.fromkeys(x for x in lang if x and x.lower() not in {"unqualified", "none"}))
    if lang:
        parts.append(f"Language: {_clip(', '.join(lang), 80)}.")
    if desc:
        parts.append(desc)
    return {
        "source_id": d["parent_asin"],
        "title": display,
        "year": year,
        "genres": ", ".join(genres),
        "fmt": fmt,
        "has_title": bool(title),
        "has_desc": bool(desc),
        "text": " ".join(parts),
    }


def build_amazon(paths: dict, report: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    r = pl.read_csv(
        paths["benchmark/5core/rating_only/Movies_and_TV.csv"],
        columns=["user_id", "parent_asin", "timestamp"],
        schema_overrides={"timestamp": pl.Int64},
    ).filter(pl.col("user_id").is_not_null() & pl.col("parent_asin").is_not_null())
    n_raw = r.height
    r = r.sort("timestamp", "parent_asin").unique(["user_id", "parent_asin"], keep="first", maintain_order=True)
    asins = set(r["parent_asin"].unique().to_list())
    rows = {}
    with open(paths["raw/meta_categories/meta_Movies_and_TV.jsonl"], "rb") as f:
        for line in f:
            d = json.loads(line)
            a = d.get("parent_asin")
            if a in asins and a not in rows:
                rows[a] = az_item(d)
    missing = sorted(asins - rows.keys())
    for a in missing:  # keep the item (it has interactions) with a minimal text
        rows[a] = az_item({"parent_asin": a})
    recs = [rows[a] for a in sorted(rows)]
    items = pl.DataFrame({
        "domain": pl.Series([AZ] * len(recs), dtype=pl.Int8),
        "source_id": [x["source_id"] for x in recs],
        "title": [x["title"] for x in recs],
        "year": pl.Series([x["year"] for x in recs], dtype=pl.Int32),
        "genres": [x["genres"] for x in recs],
        "fmt": [x["fmt"] for x in recs],
        "has_title": [x["has_title"] for x in recs],
        "tmdb_id": pl.Series([None] * len(recs), dtype=pl.Int64),
        "language": pl.Series([None] * len(recs), dtype=pl.Utf8),
        "text": [x["text"] for x in recs],
    })
    report["amazon"] = {
        "events_raw": n_raw,
        "duplicates_removed": n_raw - r.height,
        "events": r.height,
        "users": r["user_id"].n_unique(),
        "items": items.height,
        "meta_coverage": round(1 - len(missing) / len(asins), 4),
        "items_with_title": int(items["has_title"].sum()),
        "items_with_description": sum(x["has_desc"] for x in recs),
        "format_counts": dict(items["fmt"].value_counts().sort("count", descending=True).iter_rows()),
        "prime_video_items_without_title": int(items.filter((pl.col("fmt") == "Prime Video") & ~pl.col("has_title")).height),
        "top_genres": _top_values(items["genres"].str.split(", ").explode(), 15),
        "release_decades": _decades(items["year"]),
        "time_range_utc": [
            str(pl.from_epoch(pl.Series([r["timestamp"].min()]), time_unit="ms")[0]),
            str(pl.from_epoch(pl.Series([r["timestamp"].max()]), time_unit="ms")[0]),
        ],
    }
    events = r.select(
        pl.col("user_id").alias("user"), pl.col("parent_asin").alias("source_id"),
        (pl.col("timestamp") // 1000).alias("ts"),
    )
    return items, events


def _top_values(s: pl.Series, k: int) -> dict:
    s = s.drop_nulls()
    s = s.filter(s != "")
    return dict(s.value_counts().sort("count", descending=True).head(k).iter_rows())


def _decades(years: pl.Series) -> dict:
    y = years.drop_nulls()
    y = y.filter((y >= 1870) & (y <= 2030))
    return {f"{int(k)}s": int(v) for k, v in (y // 10 * 10).value_counts().sort(y.name).iter_rows()}


# ---------------------------------------------------------------------------- sequences + split
def sample_users(events: pl.DataFrame, n: int, seed: int = 0) -> pl.DataFrame:
    users = np.array(sorted(events["user"].unique().to_list()))
    keep = np.random.default_rng(seed).choice(users, size=min(n, len(users)), replace=False)
    return events.filter(pl.col("user").is_in(keep.tolist()))


def build_sequences(items: pl.DataFrame, events_by_domain: dict[int, pl.DataFrame]) -> dict:
    """Map events to item_idx and lay them out as one flat, time-ordered array per user.

    Order inside a user: timestamp, then item_idx -- deterministic even when many events
    share a timestamp (common on MovieLens sign-up).
    """
    key = items.select("domain", "source_id", "item_idx")
    parts = []
    for dom, ev in events_by_domain.items():
        e = ev.with_columns(pl.lit(dom, dtype=pl.Int8).alias("domain")).join(
            key, on=["domain", "source_id"], how="left")
        if e["item_idx"].null_count():
            raise RuntimeError(f"{e['item_idx'].null_count()} {DOMAINS[dom]} events reference unknown items")
        n_user = e.group_by("user").len()
        e = e.join(n_user.filter(pl.col("len") >= MIN_EVENTS).select("user"), on="user", how="semi")
        parts.append(e.select("domain", "user", "ts", "item_idx"))
    ev = pl.concat(parts).sort("domain", "user", "ts", "item_idx")
    lens = ev.group_by("domain", "user", maintain_order=True).len()
    offsets = np.zeros(lens.height + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens["len"].to_numpy())
    return {
        "items": ev["item_idx"].to_numpy().astype(np.int32),
        "offsets": offsets,
        "domain": lens["domain"].to_numpy().astype(np.int8),
    }


def split_stats(seq: dict, items: pl.DataFrame) -> dict:
    flat, off, dom = seq["items"], seq["offsets"], seq["domain"]
    lens = np.diff(off)
    # training portion = everything except each user's last two events
    train_mask = np.ones(len(flat), dtype=bool)
    train_mask[off[1:] - 1] = False
    train_mask[off[1:] - 2] = False
    train_counts = np.bincount(flat[train_mask], minlength=items.height + 1)
    out = {}
    for d, name in enumerate(DOMAINS):
        u = np.flatnonzero(dom == d)
        if len(u) == 0:
            continue
        test_t = flat[off[u + 1] - 1]
        val_t = flat[off[u + 1] - 2]
        dl = lens[u]
        n_items_dom = int((items["domain"] == d).sum())
        out[name] = {
            "users": int(len(u)),
            "events": int(dl.sum()),
            "train_events": int((dl - 2).sum()),
            "catalog_items": n_items_dom,
            "items_with_train_events": int((train_counts[items.filter(pl.col("domain") == d)["item_idx"].to_numpy()] > 0).sum()),
            "density": float(dl.sum() / (len(u) * n_items_dom)),
            "history_len_percentiles": {f"p{p}": float(np.percentile(dl, p)) for p in (5, 25, 50, 75, 95, 99)},
            "history_len_max": int(dl.max()),
            "cold_val_targets": int((train_counts[val_t] == 0).sum()),
            "cold_test_targets": int((train_counts[test_t] == 0).sum()),
        }
    return out


def split_hash(seq: dict) -> str:
    h = hashlib.sha256()
    for k in ("items", "offsets", "domain"):
        h.update(np.ascontiguousarray(seq[k]).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-users", type=int, default=None, help="keep N random users per domain")
    p.add_argument("--report", type=Path, default=None, help="data report path (default: results/ for full runs)")
    args = p.parse_args()

    out = processed_dir(args.sample_users)
    out.mkdir(parents=True, exist_ok=True)
    paths, provenance = fetch_all()
    report: dict = {"sources": provenance}

    print("[prepare] MovieLens-32M + TMDB", flush=True)
    ml_items, ml_events = build_movielens(paths["ml32m"], paths["tmdb"]["TMDB_movie_dataset_v11.csv"], report)
    print("[prepare] Amazon Movies & TV (5-core)", flush=True)
    az_items, az_events = build_amazon(paths["amazon"], report)

    if args.sample_users:
        ml_events = sample_users(ml_events, args.sample_users)
        az_events = sample_users(az_events, args.sample_users)
        # smaller catalog: only items the sampled users touched
        ml_items = ml_items.join(ml_events.select("source_id").unique(), on="source_id", how="semi")
        az_items = az_items.join(az_events.select("source_id").unique(), on="source_id", how="semi")
        report["sample_users_per_domain"] = args.sample_users

    # Catalog order: ML by numeric movieId, then Amazon by ASIN -> contiguous domain ranges.
    ml_items = ml_items.sort(pl.col("source_id").cast(pl.Int64))
    az_items = az_items.sort("source_id")
    items = pl.concat([ml_items, az_items]).with_row_index("item_idx", offset=1).with_columns(
        pl.col("item_idx").cast(pl.Int32))
    ranges = []
    for d in range(len(DOMAINS)):
        idx = items.filter(pl.col("domain") == d)["item_idx"]
        ranges.append([int(idx.min()), int(idx.max()) + 1])  # [lo, hi)

    print("[prepare] sequences + leave-last-out split", flush=True)
    seq = build_sequences(items, {ML: ml_events, AZ: az_events})
    report["split"] = {
        "protocol": "per-user leave-last-out by timestamp (ties broken by item index); "
                    "test = last event, validation = second-to-last, train = the rest",
        "min_events_per_user": MIN_EVENTS,
        "per_domain": split_stats(seq, items),
        "sha256": split_hash(seq),
    }
    report["catalog"] = {"n_items": items.height, "domain_ranges": dict(zip(DOMAINS, ranges))}

    items.write_parquet(out / "items.parquet")
    np.savez(out / "sequences.npz", **seq)
    meta = {"n_items": items.height, "domains": list(DOMAINS), "domain_ranges": ranges,
            "split_sha256": report["split"]["sha256"], "sample_users": args.sample_users}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    report_path = args.report or (RESULTS_DIR / "data_report.json" if not args.sample_users else out / "data_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    for name, s in report["split"]["per_domain"].items():
        print(f"[prepare] {name:9s} users={s['users']:,} events={s['events']:,} items={s['catalog_items']:,} "
              f"cold_test={s['cold_test_targets']:,}")
    print(f"[prepare] catalog={items.height:,} ranges={ranges} split_sha256={report['split']['sha256'][:16]}")
    print(f"[prepare] wrote {out} and {report_path}")


if __name__ == "__main__":
    main()

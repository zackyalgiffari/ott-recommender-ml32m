"""Download the pinned source files from the Hugging Face Hub and verify MovieLens integrity.

    python -m src.download
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.config import SOURCES


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(name: str) -> dict[str, Path]:
    """Download (or reuse from the HF cache) every file of one source."""
    src = SOURCES[name]
    return {
        fname: Path(hf_hub_download(src["repo_id"], fname, repo_type="dataset", revision=src["revision"]))
        for fname in src["files"]
    }


def verify_ml32m(paths: dict[str, Path]) -> dict[str, dict]:
    """Compare the MD5 of each MovieLens CSV against the official checksums.txt."""
    expected = {}
    for line in paths["checksums.txt"].read_text().splitlines():
        if line.strip():
            digest, fname = line.split()
            expected[fname] = digest
    report = {}
    for fname, path in paths.items():
        if fname in expected:
            got = md5sum(path)
            report[fname] = {"md5": got, "expected": expected[fname], "ok": got == expected[fname]}
    bad = [f for f, r in report.items() if not r["ok"]]
    if bad:
        raise RuntimeError(f"MD5 mismatch for {bad}; the mirror differs from the official release")
    return report


def fetch_all() -> tuple[dict[str, dict[str, Path]], dict]:
    paths = {name: fetch(name) for name in SOURCES}
    provenance = {
        name: {"repo_id": s["repo_id"], "revision": s["revision"], "files": s["files"]}
        for name, s in SOURCES.items()
    }
    provenance["ml32m"]["md5"] = verify_ml32m(paths["ml32m"])
    return paths, provenance


if __name__ == "__main__":
    paths, prov = fetch_all()
    for name, files in paths.items():
        for fname, p in files.items():
            print(f"{name:7s} {p.stat().st_size / 1e6:9.1f} MB  {fname}")
    print("ml32m md5:", {f: r["ok"] for f, r in prov["ml32m"]["md5"].items()})

"""Training windows, evaluation inputs and seen-item lists built from sequences.npz.

Layout (see prepare.py): ``items`` is one flat array of item_idx, time-ordered within each
user; user u owns ``items[offsets[u]:offsets[u+1]]``. The last event of every user is the
test target, the second-to-last the validation target, everything before is training data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sequences:
    items: np.ndarray    # int32 (E,)
    offsets: np.ndarray  # int64 (U + 1,)
    domain: np.ndarray   # int8  (U,)
    meta: dict

    @classmethod
    def load(cls, processed: Path) -> "Sequences":
        z = np.load(processed / "sequences.npz")
        meta = json.loads((processed / "meta.json").read_text())
        return cls(z["items"], z["offsets"], z["domain"], meta)

    def users(self, domain: int) -> np.ndarray:
        return np.flatnonzero(self.domain == domain)

    def train_counts(self) -> np.ndarray:
        """Occurrences of every item_idx in the training portion (popularity baseline, cold slice)."""
        mask = np.ones(len(self.items), dtype=bool)
        mask[self.offsets[1:] - 1] = False
        mask[self.offsets[1:] - 2] = False
        return np.bincount(self.items[mask], minlength=self.meta["n_items"] + 1)


def train_windows(seq: Sequences, domain: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut every user's training portion into windows of at most ``max_len + 1`` items.

    Windows are taken from the end and overlap by exactly one item, so every training
    target (each item after the first) is predicted exactly once per epoch.
    Returns (starts, ends) as absolute positions into ``seq.items``; window = items[start:end].
    """
    u = seq.users(domain)
    begin = seq.offsets[u]
    m = seq.offsets[u + 1] - 2 - begin          # training items per user
    ok = m >= 2                                  # need at least one (input, target) pair
    begin, m = begin[ok], m[ok]
    n_win = (m - 1 + max_len - 1) // max_len     # ceil((m - 1) / max_len)
    rep_begin = np.repeat(begin, n_win)
    rep_m = np.repeat(m, n_win)
    j = np.arange(n_win.sum()) - np.repeat(np.cumsum(n_win) - n_win, n_win)
    ends = rep_begin + rep_m - j * max_len
    starts = np.maximum(rep_begin, ends - max_len - 1)
    return starts.astype(np.int64), ends.astype(np.int64)


def gather_right_aligned(items: np.ndarray, starts: np.ndarray, ends: np.ndarray, length: int) -> np.ndarray:
    """(B, length) matrix: items[start:end] right-aligned (left-padded with 0), truncated to the last ``length``."""
    pos = ends[:, None] - length + np.arange(length)[None, :]
    valid = pos >= np.maximum(starts, ends - length)[:, None]
    return np.where(valid, items[np.clip(pos, 0, len(items) - 1)], 0).astype(np.int64)


def train_batch(seq: Sequences, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inputs and next-item targets for a batch of windows (dynamic padding to the batch max)."""
    length = int((ends - starts).max())
    w = gather_right_aligned(seq.items, starts, ends, length)
    return w[:, :-1], w[:, 1:]


def eval_inputs(seq: Sequences, users: np.ndarray, split: str, max_len: int):
    """Inputs, targets and history ranges for 'val' (predict 2nd-to-last) or 'test' (predict last)."""
    shift = {"val": 2, "test": 1}[split]
    first = seq.offsets[users]
    end = seq.offsets[users + 1] - shift          # target position; history = items[first:end]
    targets = seq.items[end].astype(np.int64)
    inputs = gather_right_aligned(seq.items, first, end, max_len)
    return inputs, targets, first, end


def history_index(items: np.ndarray, first: np.ndarray, end: np.ndarray, lo: int):
    """(rows, cols) of every already-seen item, with cols relative to the domain range start ``lo``."""
    lens = end - first
    rows = np.repeat(np.arange(len(first)), lens)
    pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(first, lens)
    return rows, items[pos].astype(np.int64) - lo

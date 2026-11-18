import numpy as np

from src.data import Sequences, eval_inputs, history_index, train_batch, train_windows


def test_windows_cover_each_training_target_once(toy_processed):
    seq = Sequences.load(toy_processed)
    for max_len in (1, 2, 3, 10):
        starts, ends = train_windows(seq, 0, max_len)
        assert ((ends - starts) <= max_len + 1).all()
        inp, tgt = train_batch(seq, starts, ends)
        targets = sorted(int(t) for t in tgt[(inp != 0) & (tgt != 0)])
        # user 0 trains on [1,2,3,4] -> targets 2,3,4 ; user 1 trains on [3] -> no pair
        assert targets == [2, 3, 4], (max_len, targets)


def test_windows_never_touch_val_or_test(toy_processed):
    seq = Sequences.load(toy_processed)
    for d in (0, 1):
        starts, ends = train_windows(seq, d, 2)
        last_allowed = {int(seq.offsets[u + 1] - 2) for u in seq.users(d)}
        for e in ends:
            assert any(e <= a for a in last_allowed)
        held = set(np.concatenate([seq.offsets[1:] - 1, seq.offsets[1:] - 2]).tolist())
        for s, e in zip(starts, ends):
            assert not held & set(range(int(s), int(e)))


def test_eval_inputs_targets_and_history(toy_processed):
    seq = Sequences.load(toy_processed)
    users = seq.users(0)
    inp, tgt, first, end = eval_inputs(seq, users, "test", 4)
    assert tgt.tolist() == [6, 5]
    assert inp[0].tolist() == [2, 3, 4, 5]        # last 4 before the target, right-aligned
    assert inp[1].tolist() == [0, 0, 3, 1]        # left-padded
    inp, tgt, first, end = eval_inputs(seq, users, "val", 10)
    assert tgt.tolist() == [5, 1]
    rows, cols = history_index(seq.items, first, end, lo=1)
    assert sorted(zip(rows.tolist(), (cols + 1).tolist())) == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3)]


def test_train_counts_exclude_held_out(toy_processed):
    seq = Sequences.load(toy_processed)
    c = seq.train_counts()
    assert c[5] == 0 and c[6] == 0                # user 0's val/test targets
    assert c[1] == 1 and c[3] == 2

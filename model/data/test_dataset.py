"""Torch/PIL-free verification of dataset.py's manifest parsing and tuple
alignment. Fabricates a tiny manifest + .npy frames in a temp dir (there is no
generated sim data with gl on this machine) and asserts that item k's targets
are exactly samples[ctx_end+1 .. ctx_end+H], per architecture.md §5.

Run: python3 model/data/test_dataset.py
"""

import json
import os
import sys
import tempfile

import numpy as np

# Run the same way everywhere: `python3 model/data/test_dataset.py` from the
# repo root. Inject the repo root so absolute `model.` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model.data import dataset as ds  # noqa: E402


def _fabricate(tmp, n_samples, size=8, with_frames=True):
    """A manifest whose frame i is a constant image of value i/255 and whose
    state encodes i in x, so alignment is checkable by reading the payload back."""
    frames_dir = os.path.join(tmp, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    samples = []
    for i in range(n_samples):
        rel = f"frames/{i:06d}.npy"
        if with_frames:
            np.save(os.path.join(tmp, rel), np.full((size, size, 3), i, dtype=np.uint8))
        action = None if i == 0 else {
            "steer": -1.0 + 2.0 * (i % 3) / 2.0, "throttle": 0.8,
        }
        # New sim nests the car pose under state.car; x encodes the sample index.
        s = {"action": action,
             "state": {"car": {"x": float(i), "z": 0.0, "heading": 0.0, "speed": 0.3}}}
        if with_frames:
            s = {"frame": rel, **s}
        samples.append(s)
    manifest = {"seed": 1, "steps": n_samples - 1, "dt": 1 / 30,
                "resolution": [size, size], "params": {}, "samples": samples}
    if with_frames:
        manifest["resolution"] = [size, size]
    else:
        manifest["representation"] = "state"
    path = os.path.join(tmp, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f)
    return path


def test_window_count():
    # N samples, context T, horizon H -> N-T-H+1 windows.
    assert ds.window_indices(10, 3, 2) == list(range(6))
    assert ds.window_indices(5, 3, 2) == [0]
    assert ds.window_indices(4, 3, 2) == []  # window doesn't fit
    print("ok  window count/edge cases")


def test_action_tokenizer():
    # {steer, throttle} in [-1,1] -> si*THROTTLE_BUCKETS + ti, in [0,9).
    # Neutral (straight + coast) is the center bucket (si=1,ti=1) -> 4.
    assert ds.tokenize_action(None) == ds.NEUTRAL_ACTION_TOKEN == 4
    assert ds.tokenize_action({"steer": 1.0, "throttle": 1.0}) == 8    # si=2,ti=2
    assert ds.tokenize_action({"steer": -1.0, "throttle": -1.0}) == 0  # si=0,ti=0
    assert ds.tokenize_action({"steer": 0.0, "throttle": 0.0}) == 4    # si=1,ti=1
    toks = {ds.tokenize_action({"steer": s, "throttle": t})
            for t in (-1.0, 0.0, 1.0) for s in (-1.0, 0.0, 1.0)}
    assert toks == set(range(ds.ACTION_VOCAB))  # all 9 reachable
    print("ok  action tokenizer 9-bucket range")


def test_tuple_alignment():
    context, horizon = 3, 4
    with tempfile.TemporaryDirectory() as tmp:
        path = _fabricate(tmp, n_samples=12)
        data = ds.SimSequenceDataset(path, context, horizon, representation="both", frame_size=8)
        assert len(data) == 12 - context - horizon + 1  # == 6

        for k in range(len(data)):
            item = data[k]
            cs = k  # context starts at the window index
            # target frame j must be sample (ctx_end + 1 + j) == cs + context + j,
            # and the fabricated frame's constant value encodes its sample index.
            for j in range(horizon):
                expected_idx = cs + context + j
                assert item["target_frames"][j].shape == (3, 8, 8)
                assert np.allclose(item["target_frames"][j], expected_idx / 255.0), (k, j)
                assert item["target_state"][j][0] == float(expected_idx)  # x == index
                # sample i stores the action that drives i -> i+1, so target
                # frame expected_idx must use the previous sample's action.
                assert item["target_actions"][j] == ds.tokenize_action(
                    data.manifest["samples"][expected_idx - 1]["action"]
                )
            # context frame i is sample cs+i; last context frame is right before first target.
            for i in range(context):
                assert np.allclose(item["context_frames"][i], (cs + i) / 255.0)
                frame_idx = cs + i
                expected_action = None if frame_idx == 0 else data.manifest["samples"][frame_idx - 1]["action"]
                assert item["context_actions"][i] == ds.tokenize_action(expected_action)
            assert item["meta"]["target_start"] == item["meta"]["ctx_end"] + 1
        print(f"ok  tuple alignment: {len(data)} items, targets == samples[i+1..i+H]")


def test_action_alignment_catches_one_step_shift():
    """Alternating actions make an i vs i-1 bug impossible to hide."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _fabricate(tmp, n_samples=12)
        manifest = json.load(open(path))
        for i, sample in enumerate(manifest["samples"]):
            sample["action"] = {"steer": -1.0 if i % 2 == 0 else 1.0, "throttle": 1.0}
        with open(path, "w") as f:
            json.dump(manifest, f)
        item = ds.SimSequenceDataset(path, 3, 4, representation="rgb", frame_size=8)[0]
        target_start = 3
        expected = [
            ds.tokenize_action(manifest["samples"][i - 1]["action"])
            for i in range(target_start, target_start + 4)
        ]
        shifted_wrong = [
            ds.tokenize_action(manifest["samples"][i]["action"])
            for i in range(target_start, target_start + 4)
        ]
        assert item["target_actions"].tolist() == expected
        assert expected != shifted_wrong
        print("ok  action/frame alignment uses transition i-1 -> frame i")


def test_representations():
    with tempfile.TemporaryDirectory() as tmp:
        path = _fabricate(tmp, n_samples=8)
        rgb = ds.SimSequenceDataset(path, 2, 2, representation="rgb")[0]
        assert "context_frames" in rgb and "context_state" not in rgb
        st = ds.SimSequenceDataset(path, 2, 2, representation="state")[0]
        assert "context_state" in st and "context_frames" not in st
        assert st["context_state"].shape == (2, 4)  # T x len(state)
        both = ds.SimSequenceDataset(path, 2, 2, representation="both")[0]
        assert "context_frames" in both and "context_state" in both
        print("ok  representation flag rgb|state|both")


def test_state_only_manifest():
    # A state-only export (no frames) supports 'state' but must reject 'rgb'.
    with tempfile.TemporaryDirectory() as tmp:
        path = _fabricate(tmp, n_samples=8, with_frames=False)
        st = ds.SimSequenceDataset(path, 2, 3, representation="state")
        assert st[0]["target_state"].shape == (3, 4)
        try:
            ds.SimSequenceDataset(path, 2, 3, representation="rgb")
        except ValueError:
            print("ok  state-only manifest: state works, rgb rejected")
        else:
            raise AssertionError("expected ValueError for rgb on state-only manifest")


def test_anchor_grid():
    skeleton = {
        "road": [
            {"forward": 0, "lateral": 0, "headingDelta": 0, "curvature": 0, "grade": 0},
            {"forward": 140, "lateral": 10, "headingDelta": 0.2, "curvature": 0.01, "grade": 0.02},
        ],
        "width": 9,
        "car": {"speed": 12, "slip": 0.1, "grounded": True},
        "env": {"fog": 0.4, "rain": 0.2, "snow": 0, "timeOfDay": 1.2,
                "biomeX": 0.7, "biomeY": 0.3},
    }
    a = ds.anchor_grid(skeleton)
    assert a.shape == (ds.ANCHOR_CHANNELS, 16, 16)
    assert np.isfinite(a).all()
    assert a[0].sum() > 0 and a[1].sum() == 2
    assert np.allclose(a[9], 0.4)
    print("ok  skeleton -> 16x16 anchor grid")


if __name__ == "__main__":
    test_window_count()
    test_action_tokenizer()
    test_tuple_alignment()
    test_action_alignment_catches_one_step_shift()
    test_representations()
    test_state_only_manifest()
    test_anchor_grid()
    print("\nall dataset tests passed")

"""Ingest a screen recording of the real Slow Roads game as a reference dataset.

slowroads.io blocks automated browsers, so reference footage is recorded by a
human playing normally (QuickTime / OBS). This turns that video into the same
manifest format the sim's data-gen writes (see sim/headless/generate_pixels.mjs),
so model/data/dataset.py and eval/fidelity.py load it unchanged:

    <out>/manifest.json          seed=None, dt=1/30, representation="rgb", source=...
    <out>/frames/000000.npy      (3, size, size) float32 in [0, 1]
    <out>/full/000000.png        (ref, ref) RGB, full-res copy for fidelity metrics

Frames are a centre SQUARE crop of the game viewport: the sim's capture() renders
at aspect 1 with the same vertical FOV, so a square crop is the comparable view.
Actions are unknown (null -> neutral token in the dataset); this data is a
fidelity reference / tokenizer data, not action-conditioned training data.

Clean-room note: only pixels the game displays are used. Nothing here reads,
copies or depends on Slow Roads code or assets.

Usage:
    python -m eval.ingest_recording drive.mov --out data/reference/drive1 \
        [--crop W:H:X:Y] [--start 5] [--duration 120] [--size 64] [--ref 512]

--crop selects the game viewport inside the recording (pixels of the source
video); by default ffmpeg's cropdetect trims black borders and the whole frame
is otherwise assumed to be the game. Hide the game UI before recording.
"""

import argparse
import json
import os
import re
import subprocess

try:
    import numpy as np
except ImportError:
    raise SystemExit("numpy not found: run with ~/anaconda3/bin/python (or pip install numpy pillow)")

FPS = 30  # sim.dt = 1/30; one sample per tick


def detect_crop(path, start, probe_s=10):
    """ffmpeg cropdetect over a few seconds -> 'W:H:X:Y' (black-border trim)."""
    cmd = ["ffmpeg", "-hide_banner", "-ss", str(start), "-t", str(probe_s), "-i", path,
           "-vf", "cropdetect=24:2:0", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", err)
    if not crops:
        raise RuntimeError(f"cropdetect found nothing in {path}; pass --crop W:H:X:Y")
    return max(set(crops), key=crops.count)  # most frequent, robust to fades


def ingest(path, out, crop=None, start=0.0, duration=None, size=64, ref=512):
    if not os.path.isfile(path):
        raise SystemExit(f"no such video: {path} (pass the path to your screen recording)")
    if ref % size:
        raise ValueError(f"--ref {ref} must be a multiple of --size {size} (exact box downsample)")
    crop = crop or detect_crop(path, start)
    # viewport crop -> centre square -> 30 fps -> ref x ref (area filter)
    vf = (f"crop={crop},crop='min(iw,ih)':'min(iw,ih)',fps={FPS},"
          f"scale={ref}:{ref}:flags=area,format=rgb24")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    from PIL import Image  # full-res PNG copies

    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    os.makedirs(os.path.join(out, "full"), exist_ok=True)
    frame_bytes = ref * ref * 3
    k = ref // size
    samples = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            hwc = np.frombuffer(buf, np.uint8).reshape(ref, ref, 3)
            i = len(samples)
            name = f"{i:06d}"
            Image.fromarray(hwc).save(os.path.join(out, "full", name + ".png"))
            small = hwc.reshape(size, k, size, k, 3).mean(axis=(1, 3)) / 255.0
            np.save(os.path.join(out, "frames", name + ".npy"),
                    np.ascontiguousarray(small.transpose(2, 0, 1), dtype=np.float32))
            samples.append({"frame": os.path.join("frames", name + ".npy"), "action": None})
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}) on {path}")
    if not samples:
        raise RuntimeError(f"no frames decoded from {path}")

    manifest = {
        "seed": None, "steps": len(samples) - 1, "dt": 1.0 / FPS,
        "resolution": [size, size], "representation": "rgb",
        "source": "slowroads.io screen recording",
        "reference": {"video": os.path.basename(path), "crop": crop, "start": start,
                      "duration": duration, "full_res": ref, "full_dir": "full",
                      "actions": "unknown"},
        "samples": samples,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", help="game viewport in source pixels, W:H:X:Y")
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip (menus)")
    ap.add_argument("--duration", type=float, help="seconds to keep")
    ap.add_argument("--size", type=int, default=64, help="dataset frame size")
    ap.add_argument("--ref", type=int, default=512, help="full-res PNG size")
    a = ap.parse_args()
    m = ingest(a.video, a.out, a.crop, a.start, a.duration, a.size, a.ref)
    print(f"Wrote {len(m['samples'])} frames ({len(m['samples']) / FPS:.1f}s) to {a.out} "
          f"crop={m['reference']['crop']}")


if __name__ == "__main__":
    main()

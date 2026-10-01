"""Export the Step-1 latency-spike models to ONNX for web/bench.

    export/.venv/bin/python export/export_spike.py --sweep core   # see SWEEPS
    export/.venv/bin/python export/export_spike.py --dyn 384,8,8,16,full --fp16

Writes web/bench/models/<name>.onnx (+ fp16 twin with --fp16) and merges an
entry per model into web/bench/models/manifest.json, which the bench page reads.
Weights are random: only shapes/op mix matter for latency.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(__file__))
from spike_models import FrameDynamics, FSQDecoder, n_params  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "web", "bench", "models")

# (d_model, layers, ctx_frames, grid, head)
SWEEPS = {
    "core": [(d, L, c, 16, "full") for d in (256, 384, 512) for L in (4, 8, 12) for c in (4, 8)],
    "grid32": [(d, L, c, 32, "full") for d in (256, 384) for L in (4, 8) for c in (2, 4)],
    "head": [(d, L, c, g, "fsq") for d, L, c, g in
             [(256, 4, 4, 16), (384, 8, 4, 16), (384, 8, 8, 16), (512, 12, 8, 16),
              (256, 4, 4, 32), (384, 8, 4, 32)]],
    "ctx": [(384, 8, c, 16, "full") for c in (1, 2, 16)],
    # per-layer overhead dominates on ORT Web, so test wide-and-shallow
    "wide": [(d, L, 4, 16, h) for d in (768, 1024) for L in (2, 4, 6) for h in ("full", "fsq")],
    "fused": [(d, L, c, 16, h + "+fused") for d, L, c in [(256, 4, 4), (384, 4, 4), (384, 6, 4), (512, 4, 4),
                                                          (512, 6, 4), (768, 4, 4), (384, 8, 4), (512, 4, 8)]
              for h in ("full", "fsq")],
}
DECODERS = [(128, 16, 64), (128, 16, 128), (128, 32, 128), (64, 16, 64)]  # hidden, grid, frame


def dyn_name(d, L, c, g, head):
    return f"dyn_d{d}_L{L}_c{c}_g{g}_{head.replace('+', '_')}"   # head may carry "+fused"


def to_fp16(module, args, path, keep_fp32=(), **kw):
    """Re-export the module in half precision. All float I/O (incl. the KV cache)
    becomes fp16, halving cache traffic; integer token/id I/O is unchanged."""
    out = path.replace(".onnx", "_fp16.onnx")
    m16 = module.half()
    a16 = tuple(a.half() if a.is_floating_point() and i not in keep_fp32 else a for i, a in enumerate(args))
    with torch.no_grad():
        torch.onnx.export(m16, a16, out, dynamo=False, opset_version=17, **kw)
    module.float()
    return out


def export_dyn(d, L, c, g, head, fp16):
    torch.manual_seed(0)
    fused = head.endswith("+fused")
    m = FrameDynamics(d=d, layers=L, grid=g, head=head.split("+")[0], fused=fused).eval()
    S, H, hd = m.S, m.H, m.hd
    tokens = torch.randint(0, 12800, (1, 2 * S if fused else S))
    t = torch.tensor([float(c)])
    skel = torch.rand(1, 16, g, g)
    pk = torch.randn(L, H, c * S, hd) * 0.1
    pv = torch.randn(L, H, c * S, hd) * 0.1
    name = dyn_name(d, L, c, g, head)
    path = os.path.join(OUT, name + ".onnx")
    io = dict(input_names=["tokens", "t", "skeleton", "past_k", "past_v"],
              output_names=["ids", "conf", "state", "present_k", "present_v"])
    with torch.no_grad():
        torch.onnx.export(m, (tokens, t, skel, pk, pv), path, dynamo=False, opset_version=17, **io)
        ref = m(tokens, t, skel, pk, pv)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"tokens": tokens.numpy(), "t": t.numpy(), "skeleton": skel.numpy(),
                          "past_k": pk.numpy(), "past_v": pv.numpy()})
    assert np.allclose(got[3], ref[3].numpy(), atol=1e-4), "present_k mismatch"
    assert (got[0] == ref[0].numpy()).mean() > 0.99, "ids mismatch"
    entries = [dict(name=name, kind="dyn", file=name + ".onnx", d=d, layers=L, heads=H, hd=hd,
                    ctx=c, grid=g, S=S, head=head.split("+")[0], fused=fused, dtype="fp32", params=n_params(m),
                    bytes=os.path.getsize(path))]
    if fp16:
        p16 = to_fp16(m, (tokens, t, skel, pk, pv), path, keep_fp32=(1,), **io)
        entries.append({**entries[0], "name": name + "_fp16", "file": os.path.basename(p16),
                        "dtype": "fp16", "bytes": os.path.getsize(p16)})
    return entries


def export_dec(hidden, g, frame, fp16):
    torch.manual_seed(0)
    m = FSQDecoder(hidden=hidden, grid=g, frame=frame).eval()
    ids = torch.randint(0, 12800, (1, g * g))
    name = f"dec_h{hidden}_g{g}_f{frame}"
    path = os.path.join(OUT, name + ".onnx")
    with torch.no_grad():
        torch.onnx.export(m, (ids,), path, dynamo=False, opset_version=17,
                          input_names=["ids"], output_names=["rgb"])
        ref = m(ids).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"ids": ids.numpy()})[0]
    assert np.allclose(got, ref, atol=1e-4), "decoder mismatch"
    entries = [dict(name=name, kind="dec", file=name + ".onnx", hidden=hidden, grid=g, frame=frame,
                    dtype="fp32", params=n_params(m), bytes=os.path.getsize(path))]
    if fp16:
        p16 = to_fp16(m, (ids,), path, input_names=["ids"], output_names=["rgb"])
        entries.append({**entries[0], "name": name + "_fp16", "file": os.path.basename(p16),
                        "dtype": "fp16", "bytes": os.path.getsize(p16)})
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", default=[], choices=list(SWEEPS))
    ap.add_argument("--dyn", action="append", default=[], help="d,L,ctx,grid,head")
    ap.add_argument("--decoders", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}

    cfgs = [cfg for s in args.sweep for cfg in SWEEPS[s]]
    for spec in args.dyn:
        d, L, c, g, head = spec.split(",")
        cfgs.append((int(d), int(L), int(c), int(g), head))
    for cfg in cfgs:
        if os.path.exists(os.path.join(OUT, dyn_name(*cfg) + (".onnx"))) and dyn_name(*cfg) in manifest \
                and (not args.fp16 or dyn_name(*cfg) + "_fp16" in manifest):
            continue
        for e in export_dyn(*cfg, args.fp16):
            manifest[e["name"]] = e
            print(f"{e['name']:40s} {e['params'] / 1e6:6.1f}M  {e['bytes'] / 1e6:6.1f}MB", flush=True)
        json.dump(manifest, open(man_path, "w"), indent=1)
    if args.decoders:
        for cfg in DECODERS:
            for e in export_dec(*cfg, args.fp16):
                manifest[e["name"]] = e
                print(f"{e['name']:40s} {e['params'] / 1e6:6.1f}M  {e['bytes'] / 1e6:6.1f}MB", flush=True)
        json.dump(manifest, open(man_path, "w"), indent=1)


if __name__ == "__main__":
    main()

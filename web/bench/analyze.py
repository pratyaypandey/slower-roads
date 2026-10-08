"""Summarize bench results into the Step-1 tables (markdown on stdout).

    ~/anaconda3/bin/python web/bench/analyze.py web/bench/results/m3max_*.json

Frame model (all graph-captured, fp16 unless noted):
  unfused: frame_ms(P) = P * pass + commit + decode
  fused:   frame_ms(P) = P * pass + decode    (commit rides in the next frame's first pass)
plus a few hundred µs of JS MaskGIT bookkeeping, which the measured frame loops
include; they are printed next to the model so the extrapolation can be checked.
"""

import json
import sys
from collections import defaultdict

BUDGET, HEADROOM = 33.3, 25.0   # 30 fps; ~25% headroom for sim, input, compositing, weaker-than-test GPU


def load(paths):
    rows = []
    for p in paths:
        doc = json.load(open(p))
        for r in (doc["results"] if isinstance(doc, dict) else doc):
            if r.get("error") or "probe" in r.get("job", {}):
                continue
            rows.append(r)
    return rows


med = lambda x: None if not x else x["median"]
fmt = lambda x, d=1: "" if x is None else f"{x:.{d}f}"


def label(m):
    return f"d{m['d']} L{m['layers']} ctx{m['ctx']} g{m['grid']} {m['head']}{' fused' if m.get('fused') else ''}"


def core_params(m):
    """Transformer-block params (self-attn 4d² + skeleton cross-attn 4d² + MLP 8d²
    per layer): the capacity that matters, unlike the 12800-way embed/head."""
    return 16 * m["d"] ** 2 * m["layers"]


def frame_model(r, P):
    m = r["job"]["dyn"]
    p, c, d = med(r["pass"]), med(r["commitPass"]), med(r.get("decode")) or 0
    return P * p + d + (0 if m.get("fused") else c)


def main(paths):
    rows = load(paths)
    dyn = [r for r in rows if r["job"].get("dyn")]
    decs = [r for r in rows if not r["job"].get("dyn") and r["job"].get("dec")]

    print("### Decoders (ms, ids -> RGB incl. readback)\n")
    print("| decoder | ep | kv | params M | MB | decode ms |\n|---|---|---|---|---|---|")
    for r in sorted(decs, key=lambda r: (r["job"]["dec"]["name"], r["job"].get("ep", "webgpu"), r["job"].get("kv", ""))):
        m = r["job"]["dec"]
        print(f"| {m['name']} | {r['job'].get('ep', 'webgpu')} | {r['job'].get('kv', 'gpu')} | {m['params'] / 1e6:.1f} | "
              f"{m['bytes'] / 1e6:.1f} | {fmt(med(r['decode']), 2)} |")

    print("\n### Dynamics (median ms)\n")
    print("| model | dtype | ep | kv | params M | MB | pass | commit | decode | P=1 meas/model | P=2 meas/model | P=4 meas/model |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    key = lambda r: (r["job"]["dyn"]["grid"], bool(r["job"]["dyn"].get("fused")), r["job"]["dyn"]["head"],
                     r["job"]["dyn"]["d"], r["job"]["dyn"]["layers"], r["job"]["dyn"]["ctx"],
                     r["job"]["dyn"]["dtype"], r["job"].get("ep", "webgpu"), r["job"].get("kv", "gpu"))
    for r in sorted(dyn, key=key):
        m, fr = r["job"]["dyn"], r.get("frame", {})
        cell = lambda P: f"{fmt(med(fr.get(str(P))))} / {fmt(frame_model(r, P))}" if str(P) in fr else f"— / {fmt(frame_model(r, P))}"
        print(f"| {label(m)} | {m['dtype']} | {r['job'].get('ep', 'webgpu')} | {r['job'].get('kv', 'gpu')} | "
              f"{m['params'] / 1e6:.1f} | {m['bytes'] / 1e6:.0f} | {fmt(med(r['pass']), 2)} | {fmt(med(r['commitPass']), 2)} | "
              f"{fmt(med(r.get('decode')), 2)} | {cell(1)} | {cell(2)} | {cell(4)} |")

    cap = [r for r in dyn if r["job"]["dyn"]["dtype"] == "fp16" and r["job"].get("kv") == "capture"
           and r["job"].get("ep", "webgpu") == "webgpu"]
    print(f"\n### Biggest model per budget (fp16, graph capture, 64px decode included)\n")
    print(f"| grid | P | biggest ≤ {HEADROOM:.0f} ms | frame ms | biggest ≤ {BUDGET:.0f} ms | frame ms |\n|---|---|---|---|---|---|")
    for g in sorted({r["job"]["dyn"]["grid"] for r in cap}):
        for P in (1, 2, 3, 4):
            cands = [(frame_model(r, P), r["job"]["dyn"]) for r in cap if r["job"]["dyn"]["grid"] == g]
            out = []
            for lim in (HEADROOM, BUDGET):
                ok = [c for c in cands if c[0] <= lim]
                if not ok:
                    out += ["none", ""]
                    continue
                t, m = max(ok, key=lambda c: (core_params(c[1]), -c[0]))
                out += [f"{label(m)} ({core_params(m) / 1e6:.0f}M core)", fmt(t)]
            print(f"| {g} | {P} | " + " | ".join(out) + " |")

    print("\n### Depth vs frame time (grid 16, fp16, capture; best variant per depth, P=1 and P=2)\n")
    print("| layers | width | variant | core M | P=1 ms | P=2 ms |\n|---|---|---|---|---|---|")
    for L in sorted({r["job"]["dyn"]["layers"] for r in cap}):
        for d in sorted({r["job"]["dyn"]["d"] for r in cap if r["job"]["dyn"]["layers"] == L}):
            rs = [r for r in cap if r["job"]["dyn"]["layers"] == L and r["job"]["dyn"]["d"] == d
                  and r["job"]["dyn"]["grid"] == 16 and r["job"]["dyn"]["ctx"] == 4]
            if not rs:
                continue
            b = min(rs, key=lambda r: frame_model(r, 1))
            m = b["job"]["dyn"]
            print(f"| {L} | {d} | {m['head']}{' fused' if m.get('fused') else ''} | {core_params(m) / 1e6:.1f} | "
                  f"{fmt(frame_model(b, 1))} | {fmt(frame_model(b, 2))} |")

    sus = [r for r in dyn if r.get("sustain")]
    for r in sus:
        s = r["sustain"]
        print(f"\n### Sustained {s['sec']} s, {label(r['job']['dyn'])} P={s['P']}\n")
        print(f"median {s['all']['median']:.1f} ms, p90 {s['all']['p90']:.1f}, frames {s['all']['n']}; "
              f"5 s bucket medians: {', '.join(f'{b['median']:.1f}' for b in s['buckets'])}")


if __name__ == "__main__":
    main(sys.argv[1:])

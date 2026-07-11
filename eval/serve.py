"""Tokenizer inference service — load a trained FSQ autoencoder and expose it.

The tokenizer is an encoder/decoder, so "inference" is two ops:
  encode(frame [3,64,64] float[0,1]) -> 256 int tokens
  decode(256 int tokens)            -> frame [3,64,64]
(Generative "dreaming" — tokens+action -> next tokens — arrives with the M2
dynamics model; this server already carries the codec M2 will sit on top of.)

Two ways to run it, both loading the same checkpoint via the registry:

  # 1. HTTP service (stdlib only — no fastapi/flask dep). Prompt it from anywhere:
  python -m eval.serve --ckpt fsq_v2_h128_best.pt --http --port 8000
    GET  /health                      -> {arch, tokens_per_frame, codebook_size, ...}
    POST /encode   {"frames_b64":..}  -> {"tokens": [[...256...], ...]}
    POST /decode   {"tokens": [[..]]} -> {"frames_b64": ..}
    POST /roundtrip{"frames_b64":..}  -> {"frames_b64":.., "l1": ..}
  Arrays cross the wire as base64 of a numpy .npy buffer (encode float32
  (N,3,64,64) in [0,1]; tokens int64 (N,256)). See _to_b64 / _from_b64.

  # 2. Local self-test — encodes real frames, round-trips, writes a compare PNG:
  python -m eval.serve --ckpt fsq_v2_h128_best.pt --demo --data data/seed1
"""

import argparse
import base64
import io
import json
import os

import numpy as np
import torch

from model.registry import load_tokenizer


def _to_b64(arr):
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _from_b64(s):
    return np.load(io.BytesIO(base64.b64decode(s)))


class Tokenizer:
    """Thin wrapper: batched encode/decode on the loaded checkpoint."""

    def __init__(self, ckpt, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.meta = load_tokenizer(ckpt, map_location=self.device)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def info(self):
        m = self.model
        return {"arch": self.meta.get("builder"), "cfg": self.meta.get("cfg"),
                "tokens_per_frame": int(m.tokens_per_frame),
                "codebook_size": int(m.codebook_size),
                "device": str(self.device)}

    @torch.no_grad()
    def encode(self, frames):
        x = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        _, idx, _ = self.model(x)                 # forward returns (recon, indices, ...)
        return idx.cpu().numpy()                  # (N, tokens)

    @torch.no_grad()
    def decode(self, tokens):
        idx = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        return self.model.decode_indices(idx).cpu().numpy()   # (N,3,64,64)

    @torch.no_grad()
    def roundtrip(self, frames):
        x = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        recon, _, _ = self.model(x)
        l1 = (recon - x).abs().mean().item()
        return recon.cpu().numpy(), l1


def _make_handler(tok):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, **tok.info()})
            else:
                self._send(404, {"error": "GET /health"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                path = self.path.rstrip("/")
                if path == "/encode":
                    toks = tok.encode(_from_b64(req["frames_b64"]))
                    self._send(200, {"tokens": toks.tolist()})
                elif path == "/decode":
                    frames = tok.decode(np.asarray(req["tokens"], dtype=np.int64))
                    self._send(200, {"frames_b64": _to_b64(frames.astype(np.float32))})
                elif path == "/roundtrip":
                    recon, l1 = tok.roundtrip(_from_b64(req["frames_b64"]))
                    self._send(200, {"frames_b64": _to_b64(recon.astype(np.float32)), "l1": l1})
                else:
                    self._send(404, {"error": "POST /encode|/decode|/roundtrip"})
            except Exception as e:  # noqa: BLE001 — return the error to the caller
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):  # quiet
            pass

    return H


def demo(tok, data_dir, n=6, out="eval/serve_demo.png"):
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    idxs = np.linspace(0, len(manifest["samples"]) - 1, n).astype(int)
    frames = np.stack([np.load(os.path.join(data_dir, manifest["samples"][i]["frame"])) for i in idxs])
    toks = tok.encode(frames)
    recon, l1 = tok.roundtrip(frames)
    print(json.dumps({**tok.info(), "demo_frames": n, "roundtrip_l1": round(l1, 5),
                      "tokens_shape": list(toks.shape)}, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, n, figsize=(2 * n, 4.2))
        for j in range(n):
            ax[0, j].imshow(frames[j].transpose(1, 2, 0).clip(0, 1)); ax[0, j].axis("off")
            ax[1, j].imshow(recon[j].transpose(1, 2, 0).clip(0, 1)); ax[1, j].axis("off")
        ax[0, 0].set_title("orig", loc="left"); ax[1, 0].set_title("decode", loc="left")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=110); print(f"saved {out}")
    except ImportError:
        print("(matplotlib absent — numbers above)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--http", action="store_true", help="run the HTTP service")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--demo", action="store_true", help="local round-trip self-test")
    ap.add_argument("--data", default="data/seed1")
    args = ap.parse_args()

    tok = Tokenizer(args.ckpt, args.device)
    if args.demo:
        demo(tok, args.data)
    if args.http:
        from http.server import ThreadingHTTPServer
        srv = ThreadingHTTPServer((args.host, args.port), _make_handler(tok))
        print(f"serving tokenizer on {args.host}:{args.port}  ({tok.info()})", flush=True)
        srv.serve_forever()
    if not args.demo and not args.http:
        print(json.dumps(tok.info(), indent=1))


if __name__ == "__main__":
    main()

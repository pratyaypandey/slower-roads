"""Serve the FSQ tokenizer on Modal — the serverless counterpart to RunPod.

Unlike the RunPod flow (create pod -> SSH -> run -> remember to delete), Modal
defines the image + GPU + storage in code and runs functions on demand, scaling to
zero between calls (per-second billing, no pod to leave running).

  modal run   deploy/modal_serve.py        # one-shot GPU demo (real round-trip L1)
  modal deploy deploy/modal_serve.py        # persistent HTTPS endpoint (prints URL)
  modal app stop sr-tokenizer               # tear the endpoint down

The checkpoint + a few sample frames live on the Modal Volume `sr-models`
(uploaded with `modal volume put sr-models checkpoints/tokenizer.pt /tokenizer.pt`).
"""

import modal

app = modal.App("sr-tokenizer")

# Image is code, not a Dockerfile: base + deps + the local tokenizer package.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "fastapi[standard]")
    .add_local_python_source("model")
)
vol = modal.Volume.from_name("sr-models")


@app.cls(image=image, gpu="T4", volumes={"/models": vol}, scaledown_window=120)
class Tokenizer:
    @modal.enter()
    def load(self):
        # Runs once per container (not per request) — model stays warm for reuse.
        import torch
        from model.registry import load_tokenizer
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = load_tokenizer("/models/tokenizer.pt", map_location=self.dev)
        self.model = self.model.to(self.dev).eval()

    def _roundtrip(self, x):
        import torch
        with torch.no_grad():
            recon, idx, _ = self.model(x)
        return recon, idx

    @modal.method()
    def demo(self):
        import glob
        import numpy as np
        import torch
        fs = sorted(glob.glob("/models/frames/*.npy"))
        x = torch.from_numpy(np.stack([np.load(f) for f in fs])).float().to(self.dev)
        recon, idx = self._roundtrip(x)
        return {"device": self.dev, "n_frames": len(fs),
                "tokens_per_frame": int(self.model.tokens_per_frame),
                "codebook_size": int(self.model.codebook_size),
                "roundtrip_l1": round((recon - x).abs().mean().item(), 5),
                "tokens_shape": list(idx.shape)}

    @modal.fastapi_endpoint(method="POST")
    def roundtrip(self, item: dict):
        # item = {"frames_b64": base64(np.save float32 (N,3,64,64) in [0,1])}
        import base64
        import io
        import numpy as np
        import torch
        arr = np.load(io.BytesIO(base64.b64decode(item["frames_b64"])))
        x = torch.from_numpy(arr).float().to(self.dev)
        recon, idx = self._roundtrip(x)
        buf = io.BytesIO()
        np.save(buf, recon.cpu().numpy().astype(np.float32))
        return {"tokens": idx.cpu().numpy().tolist(),
                "frames_b64": base64.b64encode(buf.getvalue()).decode(),
                "l1": (recon - x).abs().mean().item()}


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(Tokenizer().demo.remote(), indent=1))

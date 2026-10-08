"""How good can a 64 px frame look? 64 -> 256 upscaler study.

Trains small super-resolution nets on our own sim (high-res renders from
gen_hr.mjs, box-downsampled x4 = what the 64 px pipeline sees), compares them
with classic resamplers and a pretrained generic model (Real-ESRGAN
realesr-general-x4v3), and exports the nets to ONNX for the WebGPU bench.

    export/.venv/bin/python export/sr/sr_study.py train --arch small --loss l1 --minutes 8
    export/.venv/bin/python export/sr/sr_study.py train --arch small --loss gan --init small_l1 --minutes 8
    export/.venv/bin/python export/sr/sr_study.py eval
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data", "sr_hr")
CK = os.path.join(ROOT, "checkpoints", "sr")
OUT = os.path.join(ROOT, "export", "sr", "out")
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_split(split, limit=None):
    files = sorted(glob.glob(os.path.join(DATA, split, "*", "*.png")))[:limit]
    hr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])     # (N,256,256,3) uint8
    return torch.from_numpy(hr).permute(0, 3, 1, 2).contiguous(), files


def to_lr(hr):
    """Box downsample x4 -- the capture path's own filter (supersample + box)."""
    return F.avg_pool2d(hr, 4)


# ---------------------------------------------------------------- models
class SRVGG(nn.Module):
    """SRVGGNetCompact layout (conv/PReLU stack, PixelShuffle x4, nearest skip).
    num_feat=64,num_conv=32 is exactly Real-ESRGAN realesr-general-x4v3."""

    def __init__(self, num_feat=64, num_conv=32, scale=4):
        super().__init__()
        self.scale = scale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        body += [nn.Conv2d(num_feat, 3 * scale * scale, 3, 1, 1)]
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.scale, mode="nearest")


ARCHS = {
    "tiny": dict(num_feat=24, num_conv=4),      # ~30k params
    "small": dict(num_feat=32, num_conv=8),     # ~90k
    "medium": dict(num_feat=48, num_conv=16),   # ~350k
    "large": dict(num_feat=64, num_conv=32),    # ~1.2M = Real-ESRGAN compact size
}


def build(arch):
    return SRVGG(**ARCHS[arch])


def load_realesrgan():
    m = SRVGG(64, 32)
    sd = torch.load(os.path.join(CK, "realesr-general-x4v3.pth"), map_location="cpu")
    m.load_state_dict(sd.get("params", sd.get("params_ema", sd)))
    return m.eval()


class PatchD(nn.Module):
    """Small PatchGAN discriminator (spectral norm)."""

    def __init__(self, ch=48):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, c = [], 3
        for i, co in enumerate([ch, ch * 2, ch * 4, ch * 4]):
            layers += [sn(nn.Conv2d(c, co, 4, 2 if i < 3 else 1, 1)), nn.LeakyReLU(0.2, True)]
            c = co
        layers += [sn(nn.Conv2d(c, 1, 3, 1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x * 2 - 1)


# ---------------------------------------------------------------- train
def train(args):
    hr_all, _ = load_split("train")
    print(f"train frames {len(hr_all)} on {DEV}")
    g = build(args.arch).to(DEV)
    if args.init == "realesrgan":
        g.load_state_dict(load_realesrgan().state_dict())
    elif args.init:
        g.load_state_dict(torch.load(os.path.join(CK, args.init + ".pt"), map_location="cpu"))
    opt = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.9, 0.99))
    perceptual = args.loss in ("lpips", "gan")
    if perceptual:
        import lpips
        lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval()
        for p in lp.parameters():
            p.requires_grad_(False)
    if args.loss == "gan":
        d = PatchD().to(DEV)
        opt_d = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.9, 0.99))
    P = args.patch                                # HR patch side
    t0, step, log = time.time(), 0, []
    while time.time() - t0 < args.minutes * 60:
        idx = torch.randint(0, len(hr_all), (args.batch,))
        y0, x0 = np.random.randint(0, 256 - P + 1, 2) // 4 * 4
        hr = hr_all[idx, :, y0:y0 + P, x0:x0 + P].to(DEV).float() / 255
        if np.random.rand() < 0.5:
            hr = hr.flip(-1)
        lr = to_lr(hr)
        if args.noise > 0:                         # robustness to tokenizer error
            lr = (lr + args.noise * torch.rand(len(lr), 1, 1, 1, device=DEV) * torch.randn_like(lr)).clamp(0, 1)
        sr = g(lr)
        loss = F.l1_loss(sr, hr)
        if perceptual:
            loss = loss + args.lpips_w * lp(sr * 2 - 1, hr * 2 - 1).mean()
        if args.loss == "gan":
            loss = loss + args.gan_w * F.softplus(-d(sr)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if args.loss == "gan":
            ld = F.softplus(-d(hr)).mean() + F.softplus(d(sr.detach())).mean()
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()
        step += 1
        if step % 200 == 0:
            log.append((step, float(loss)))
            print(f"step {step} loss {float(loss):.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(CK, exist_ok=True)
    name = args.name or f"{args.arch}_{args.loss}"
    torch.save(g.state_dict(), os.path.join(CK, name + ".pt"))
    print(f"saved {name} after {step} steps, params {sum(p.numel() for p in g.parameters())}")


# ---------------------------------------------------------------- eval
def pil_resize(lr_u8, mode):
    return np.stack([np.asarray(Image.fromarray(x).resize((256, 256), mode)) for x in lr_u8])


@torch.no_grad()
def run_net(m, lr):                               # lr float (N,3,64,64) in [0,1]
    m = m.to(DEV).eval()
    return torch.cat([m(lr[i:i + 32].to(DEV)).clamp(0, 1).cpu() for i in range(0, len(lr), 32)])


def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    return float((10 * torch.log10(1 / mse.clamp_min(1e-10))).mean())


def methods():
    out = {"nearest": None, "bilinear": None, "bicubic": None, "lanczos": None}
    for f in sorted(glob.glob(os.path.join(CK, "*.pt"))):
        out[os.path.basename(f)[:-3]] = f
    out["realesrgan_x4v3"] = "pretrained"
    return out


@torch.no_grad()
def evaluate(args):
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    hr_u8, files = load_split("test")
    hr_u8, files = hr_u8[:: args.every], files[:: args.every]
    ep = [os.path.dirname(f) for f in files]
    same = torch.tensor([ep[i] == ep[i - 1] for i in range(1, len(ep))])   # adjacent pairs within an episode
    hr = hr_u8.float() / 255
    lr = to_lr(hr)
    lr_u8 = (lr * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    res, outs = {}, {}
    for name, src in methods().items():
        if src is None:
            mode = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
                    "lanczos": Image.LANCZOS}[name]
            sr = torch.from_numpy(pil_resize(lr_u8, mode)).permute(0, 3, 1, 2).float() / 255
            params = 0
        else:
            if src == "pretrained":
                m = load_realesrgan()
            else:
                arch = os.path.basename(src).split("_")[0]
                m = build(arch)
                m.load_state_dict(torch.load(src, map_location="cpu"))
            sr = run_net(m, lr)
            params = sum(p.numel() for p in m.parameters())
        l = torch.cat([lp(sr[i:i + 32].to(DEV) * 2 - 1, hr[i:i + 32].to(DEV) * 2 - 1).flatten().cpu()
                       for i in range(0, len(sr), 32)])
        # flicker: how much the frame-to-frame change departs from the truth's change (x100)
        dsr, dhr = sr[1:] - sr[:-1], hr[1:] - hr[:-1]
        flick = float((dsr - dhr).abs().mean(dim=(1, 2, 3))[same].mean() * 100)
        res[name] = dict(psnr=psnr(sr, hr), lpips=float(l.mean()), flicker=flick, params=params)
        outs[name] = sr
        print(f"{name:22s} PSNR {res[name]['psnr']:.2f}  LPIPS {res[name]['lpips']:.4f}  flicker {flick:.3f}  params {params}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    torch.save({k: v[: args.keep] for k, v in outs.items()} | {"hr": hr[: args.keep]}, os.path.join(OUT, "samples.pt"))


def export_onnx(args):
    os.makedirs(os.path.join(ROOT, "web", "bench", "models"), exist_ok=True)
    man_p = os.path.join(ROOT, "web", "bench", "models", "manifest.json")
    man = json.load(open(man_p))
    for name, src in methods().items():
        if src is None:
            continue
        m = load_realesrgan() if src == "pretrained" else build(os.path.basename(src).split("_")[0])
        if src != "pretrained":
            m.load_state_dict(torch.load(src, map_location="cpu"))
        m = m.eval().half()
        fn = f"sr_{name}_fp16.onnx"
        path = os.path.join(ROOT, "web", "bench", "models", fn)
        with torch.no_grad():
            torch.onnx.export(m, (torch.rand(1, 3, 64, 64).half(),), path, dynamo=False, opset_version=17,
                              input_names=["lr"], output_names=["sr"])
        man["sr_" + name] = dict(name="sr_" + name, kind="sr", file=fn, dtype="fp16",
                                 params=sum(p.numel() for p in m.parameters()), bytes=os.path.getsize(path))
        print("exported", fn)
    json.dump(man, open(man_p, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "export"])
    ap.add_argument("--arch", default="small", choices=list(ARCHS))
    ap.add_argument("--loss", default="l1", choices=["l1", "lpips", "gan"])
    ap.add_argument("--init", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--lpips-w", type=float, default=0.5)
    ap.add_argument("--gan-w", type=float, default=0.02)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--keep", type=int, default=400)
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "export": export_onnx}[a.cmd](a)

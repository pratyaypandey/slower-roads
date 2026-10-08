"""Side-by-side sheet: 64 px input through each upscaler vs the 256 px truth.

    export/.venv/bin/python export/sr/make_sheet.py --methods nearest,bicubic,small_l1,medium_gan,large_ftgan \
        --out docs/upscale_sheet.png
Rows: held-out sim worlds (data/sr_hr/test) + real Slow Roads frames
(data/reference/drive1/full, eval only -- nothing is trained on them).
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import sr_study as S  # noqa: E402

LABEL = {"nearest": "64 px (nearest)", "bicubic": "bicubic", "lanczos": "lanczos", "bilinear": "bilinear",
         "realesrgan_x4v3": "Real-ESRGAN (generic)", "hr": "256 px truth"}


def upscale(name, lr_u8, lr):
    if name in ("nearest", "bilinear", "bicubic", "lanczos"):
        mode = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}[name]
        return S.pil_resize(lr_u8, mode)
    if name == "realesrgan_x4v3":
        m = S.load_realesrgan()
    else:
        m = S.build(name.split("_")[0])
        m.load_state_dict(torch.load(os.path.join(S.CK, name + ".pt"), map_location="cpu"))
    sr = S.run_net(m, lr)
    return (sr.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="nearest,bicubic,small_l1,medium_gan,large_ftgan,realesrgan_x4v3")
    ap.add_argument("--sim", default="s30030_cruise/0040,s30031_keys_lane/0120,s30033_dial_mix/0100")
    ap.add_argument("--real", default="003000,009300,018600")
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--out", default=os.path.join(S.ROOT, "docs", "upscale_sheet.png"))
    a = ap.parse_args()
    paths = [os.path.join(S.DATA, "test", p + ".png") for p in a.sim.split(",") if p]
    paths += [os.path.join(S.ROOT, "data", "reference", "drive1", "full", p + ".png") for p in a.real.split(",") if p]
    kinds = ["Our sim (held-out world)"] * len(a.sim.split(",")) + ["Real Slow Roads"] * len(a.real.split(","))
    hr_u8 = np.stack([np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BOX)) for p in paths])
    hr = torch.from_numpy(hr_u8).permute(0, 3, 1, 2).float() / 255
    lr = S.to_lr(hr)
    lr_u8 = (lr * 255).round().byte().permute(0, 2, 3, 1).numpy()
    methods = a.methods.split(",")
    cols = [upscale(m, lr_u8, lr) for m in methods] + [hr_u8]
    names = [LABEL.get(m, m) for m in methods] + [LABEL["hr"]]

    T, pad, top, left = a.tile, 8, 40, 170
    sheet = Image.new("RGB", (left + len(cols) * (T + pad), top + len(paths) * (T + pad)), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
    except OSError:
        f = ImageFont.load_default()
    for c, n in enumerate(names):
        d.text((left + c * (T + pad) + 6, 10), n, fill=(235, 235, 235), font=f)
    for r in range(len(paths)):
        y = top + r * (T + pad)
        d.text((10, y + T // 2 - 20), kinds[r].replace(" (", "\n("), fill=(235, 235, 235), font=f)
        for c, col in enumerate(cols):
            sheet.paste(Image.fromarray(col[r]).resize((T, T), Image.LANCZOS if c else Image.NEAREST), (left + c * (T + pad), y))
    sheet.save(a.out)
    print(a.out, sheet.size)


if __name__ == "__main__":
    main()

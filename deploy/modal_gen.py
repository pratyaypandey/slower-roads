"""Generate sim seeds on Modal — render pixel datasets in the cloud so the local
machine's CPU (and fan) stays free.

The sim renders through headless Chromium's WebGL (SwiftShader / CPU GL), driven by
Playwright, exactly like sim/headless/generate_pixels.mjs does locally. Each seed
is one container writing data/seedN/{manifest.json, frames/*.npy} to the sr-models
volume; seeds render in parallel.

  modal run deploy/modal_gen.py                       # seeds 6..12 (default), 2500 steps
  modal run deploy/modal_gen.py --seeds 6,7,8 --steps 2500

The image bakes Playwright + Chromium and the sim source. Playwright is installed
INTO /sim/node_modules because the sim does an ESM `import("playwright")`, which
resolves up the directory tree from the script — not via NODE_PATH.
"""

import modal

app = modal.App("sr-gen")

image = (
    modal.Image.from_registry("node:20-bookworm", add_python="3.11")
    .add_local_dir("sim", "/sim", copy=True, ignore=["node_modules", "test", "demo"])
    .run_commands(
        "cd /sim && npm install playwright@1.61.1",
        "cd /sim && PLAYWRIGHT_BROWSERS_PATH=/ms-playwright npx playwright install --with-deps chromium",
    )
    .env({"PLAYWRIGHT_BROWSERS_PATH": "/ms-playwright"})
)
vol = modal.Volume.from_name("sr-models")


@app.function(image=image, volumes={"/models": vol}, timeout=60 * 60, cpu=4.0)
def gen(seed: int, steps: int = 2500, size: int = 64):
    import os
    import subprocess
    out = f"/models/data/seed{seed}"
    subprocess.run(
        ["node", "headless/generate_pixels.mjs",
         "--seed", str(seed), "--steps", str(steps), "--size", str(size), "--out", out],
        cwd="/sim", check=True,
    )
    vol.commit()
    n = len(os.listdir(os.path.join(out, "frames")))
    return {"seed": seed, "frames": n}


@app.local_entrypoint()
def main(seeds: str = "6,7,8,9,10,11,12", steps: int = 2500, size: int = 64):
    seed_list = [int(s) for s in seeds.split(",")]
    print(f"generating seeds {seed_list} ({steps} steps @ {size}px) in parallel...")
    for res in gen.starmap([(s, steps, size) for s in seed_list]):
        print("done:", res)

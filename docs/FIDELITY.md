# Visual fidelity vs. Slow Roads

Goal: our sim's frames should be indistinguishable in distribution from the real
game, so the world model learns Slow Roads' look while we keep the deterministic
oracle (`sim/core`: pose, road skeleton, dials) that M3's anchor/state head needs.

## Ground rules (clean room)
Slow Roads is **not open source**. The GitHub "slowroads" repos are unofficial
scrapes of the minified production build, labelled CC BY-NC-ND 4.0 (no
derivatives). So:
- Never copy, port or consult Slow Roads code, shaders, textures or models. Our
  renderer (`sim/render/`) is rebuilt from **observation only**: recorded footage
  plus the metrics below.
- slowroads.io blocks automated browsers (Cloudflare bot check). We don't get around
  that. Reference footage is a person playing normally and screen-recording.

## 1. Record reference footage
- Play at https://slowroads.io in a normal browser window, full screen or a
  fixed window size. Use default graphics settings and the default car with the
  chase camera, and **hide the UI** so the recording is only the game view.
- Record with QuickTime (File → New Screen Recording, window or region) or OBS,
  at 30 or 60 fps.
- Drive a mix: a long straight cruise, gentle and hard steering, braking, and
  some autopilot (`F`). Capture a few 2–5 minute drives across different
  times of day and biomes. Note the settings you used.

## 2. Ingest
```bash
python -m eval.ingest_recording ~/Movies/drive1.mov --out data/reference/drive1 \
    --start 5 [--duration 180] [--crop W:H:X:Y]
```
- The output is the standard manifest (`frames/*.npy` at 64px, `full/*.png` at
  512px) with `action: null`.
- The crop is a centre square of the viewport, because our `capture()` renders at
  aspect 1 with the same vertical FOV.
- Without `--crop`, only black borders are trimmed, so crop out any browser chrome.

## 3. Measure
```bash
# ours: render at 256px (larger captures overflow the JS->Node string limit)
for s in 1 2 3; do SLOWSIM_CHANNEL=chrome SLOWSIM_GL=gpu node sim/headless/generate_pixels.mjs \
    --seed $s --steps 600 --size 256 --out data/fidelity_baseline/v2_seed$s; done
B=data/fidelity_baseline
python -m eval.fidelity --ref data/reference/drive1 \
    --ours $B/v2_seed1,$B/v2_seed2,$B/v2_seed3 --name v2 --max-frames 600
python -m eval.calibrate_view data/reference/drive1 $B/v2_seed1   # car bbox + palette
node sim/headless/snapshot.mjs --seed 2 --out snap.png             # quick-look stills
```
- Each metric is a 1-D Wasserstein distance between reference and ours:
  - per-band CIELAB colour
  - per-band edge density
  - power-spectrum slope
  - horizon height
  - frame-to-frame change
- Each distance is reported as a **ratio to a noise floor**. The default (`--floor
  clips`) is how far random real clips, shaped like ours (one per dataset, same
  length), sit from the full reference. `--floor halves` (first vs second half of a
  long drive) is much stricter and unfair to short renders.
- The run writes `eval/fidelity_report.json` and a contact sheet
  (`eval/fidelity_sheet.png`: the reference row, then one row per renderer).

**Gate:** every ratio ≤ 3× (`--gate`). Provisional; there is one reference drive so far.

## 4. Rebuild loop: what was matched, and how
The engine is `sim/render/` (renderer.js, shaders.js, textures.js, car.js). The old
v1 renderer, its post-processing passes and Sky.js were deleted once v2 beat it.

- **Colour pipeline.** All materials are custom shaders in display colours: no tone
  mapping, no post, `THREE.ColorManagement` off. v1's `capture()` skipped its
  composer and three r160 skips tone mapping/sRGB conversion for render targets, so
  v1 training frames were linear-light (washed-out sky, near-black road). Now the
  canvas and `capture()` agree, and capture supersamples, box-filters, then blurs at
  σ = 0.4 px per 256 px (the reference's measured softness; fixes spectrum slope).
- **Camera** (`CAM` in renderer.js). Solved jointly with the car's profile from the
  reference's **median car sprite**: the reference camera is rigid, so the car is a
  near-constant image. Values: vertical FOV 60°, 6.08 m back, 2.73 m up, 7.25° pitch,
  horizon ≈ 0.39; near-rigid pitch, slight yaw lag.
- **Car** (car.js). A white hatchback whose roof / rear-screen / deck-lip /
  rear-panel / bumper rows match the median sprite. The car has its own lighting:
  top surfaces clip to white, the vertical rear panel sits at ~40%.
- **Road.** Measured per-column in image space: a ~7.2 m road, the car's centre
  ~1.8 m from the left asphalt edge (the autopilot drives the left lane:
  `AUTOPILOT_LANE` in core/sim.js), white edge lines and 9 m-period centre dashes
  drawn in the fragment shader from exact road coordinates, a thin gravel shoulder,
  guardrails where the ground falls away.
- **Palette** (`PAL`). Sampled from reference pixel classes (sky, road, grass,
  foliage, rock), then desaturated greens and stronger haze to match per-band Lab
  quantiles.
- **World retune** (sim/core). Steeper road cuts (`ROLL_SPAN` 46→16 m), larger hills
  (`H_FREQ`/`H_AMP`/`H_AMP2`), smoother road profile (`GRADE_RESPONSE` 20→30; low
  gravity still launches at crests), hillside cross-slopes beside the road
  (`hillside()` in road.js), clumpier groves, fewer conifers, trees closer to the road.
  Car handling (WASD) is unchanged: the reference is autopilot footage, so it
  constrains look and camera, not driving dynamics.

## Status (3 seeds × 20 s vs the 13-minute reference, clip floor)

| metric | old v1 | v2 |
|---|---|---|
| sky L / a / b | 4.5 / 8.0 / 5.6 | **2.1 / 2.8** / 3.5 |
| sky edges | 2.0 | **0.9** |
| mid L / a / b | 8.5 / 5.5 / 9.8 | **1.4** / 3.1 / 4.9 |
| mid edges | 3.4 | 3.5 |
| ground L / a / b | 33.1 / 36.0 / 19.3 | 7.2 / 5.0 / 4.5 |
| ground edges | 37.8 | 4.6 |
| spectrum slope | 30.6 | **2.1** |
| horizon | 19.7 | 3.1 |
| temporal | 6.0 | **1.1** |

**Remaining gaps**, roughly in order of payoff:
- Ground band (the car's lower half and the road around it). The reference car is a
  rounded modern coupe; ours is boxier, especially its haunches and bumper.
- Mid-band blue-yellow and edge density. The reference's horizon zone is smoother:
  more hillside and haze, fewer silhouetted trees.
- The reference's hills are often taller and closer than ours (more terrain above
  the horizon).
- Not built yet: curve chevrons / warning signs, guardrail collision.
- Only daytime, clear, default-biome footage exists. Night, fog, rain, snow and
  biome dials are implemented but unvalidated against the game. Record more drives
  (different times of day / weather) to extend the reference and tighten the floor.

When the gate passes, regenerate the training datasets with the new renderer.
Existing tokenizer/dynamics checkpoints were trained on v1 pixels and the pre-retune
world.

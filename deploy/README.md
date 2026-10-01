# deploy/ — serving the model

Two ways to run the tokenizer (and, later, the M2 world model) as inference:

| | file | best for |
|---|---|---|
| **Modal** | `modal_serve.py` | on-demand HTTPS endpoint, scale-to-zero, per-second billing, zero pod management. `modal run` for a one-shot GPU demo, `modal deploy` for a persistent URL. Uses Modal workspace **`slower-roads`** + Volume `sr-models`. |
| **RunPod** | `RUNPOD_SERVE.md` | interactive GPU box (SSH) + cheap sustained runs. Spin up a pod attached to the network volume `sr-models` (EU-CZ-1), run `eval/serve.py`, tear down. |

Rule of thumb: **Modal for serving + bursty parallel jobs; RunPod for hands-on
experimentation + cheap long training.** Both stores hold the same
`checkpoints/tokenizer.pt`, so either path is ready.

The model-loading + encode/decode logic itself lives in `eval/serve.py` (framework-
agnostic); `modal_serve.py` wraps it for Modal, and `RUNPOD_SERVE.md` documents the
RunPod pod workflow.

## M3 workspace

M3 training/evaluation is isolated under Modal profile **`slower-roads-m3`**:

- `sr-m3-train` holds train/validation caches, model checkpoints, and metrics.
- `sr-m3-val` holds seed5 RGB oracle frames for pixel-space promotion gates.
- `sr-m3-test` holds pristine seed2 and is mounted only by `evaluate_m3`.
- `deploy/modal_train.py` defines the `sr-m3` app; Volume names can be overridden
  with `SR_MODAL_TRAIN_VOLUME` / `SR_MODAL_VAL_VOLUME` / `SR_MODAL_TEST_VOLUME`.
- `deploy/modal_gen.py` defaults to `sr-m3-train`; override with
  `SR_MODAL_VOLUME=sr-m3-val` or `sr-m3-test` for isolated oracle generation.

Smoke all M3 mechanisms on an A10:

```bash
modal profile activate slower-roads-m3
modal run deploy/modal_train.py::smoke
```

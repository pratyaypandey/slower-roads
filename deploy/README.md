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

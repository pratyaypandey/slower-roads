# Serving the tokenizer off RunPod

The trained tokenizer + code + data live on a **persistent RunPod network volume**, so
you can spin up a GPU pod, prompt the model, and tear it down — paying GPU only while
it's up (the volume itself is ~$0.70/mo idle).

## What's stored

**Network volume `sr-models`** — id `w8ha0r38ag`, datacenter **EU-CZ-1**, 10GB.
Mounts at `/workspace`. Contents under `/workspace/sr/`:

| path | what |
|---|---|
| `checkpoints/tokenizer.pt` | shipped tokenizer (fsq_v2, hidden 128, L1 ~0.0009) |
| `models/fsq_v2_h128_best.pt` | same, lean copy |
| `models/exp_L2_hidden128_winner_full.pt` | winner + optimizer state (resumable) |
| `models/exp_L1_hidden64_ema_cosine_600ep.pt` | training-only ablation (L1 0.0022) |
| `model/ eval/ headless/` | code (tokenizer, `eval/serve.py`, sim capture) |
| `data/seed1/` | 2501 eval frames + manifest |
| `VAE_RECIPE.md` | the recipe + experiment addendum |

Because the volume is in EU-CZ-1, **pods that attach it must be in EU-CZ-1** (that DC has
3090/4090). The network volume requires **secure cloud**.

## Bring it up (whenever you want)

```bash
# 1. Create a pod with the volume attached (secure cloud; key injected; auto-terminate backstop).
TERM_AT=$(date -u -v+3H +%Y-%m-%dT%H:%M:%SZ)
ENV=$(python3 -c "import json;print(json.dumps({'PUBLIC_KEY':open('$HOME/.ssh/id_ed25519.pub').read().strip()}))")
runpodctl pod create --name sr-serve --gpu-id "NVIDIA GeForce RTX 3090" \
  --image "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404" \
  --cloud-type SECURE --network-volume-id w8ha0r38ag --ports "22/tcp" \
  --container-disk-in-gb 20 --env "$ENV" --terminate-after "$TERM_AT"

# 2. Get the SSH endpoint (not shown by runpodctl — query the API; never echo the key).
KEY=$(python3 -c "import re;print(re.search(r'rpa_[A-Za-z0-9]+', open('$HOME/.runpod/config.toml').read()).group())")
POD=<pod-id-from-step-1>
curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" -H "Content-Type: application/json" \
  -d "{\"query\":\"query{pod(input:{podId:\\\"$POD\\\"}){runtime{ports{ip isIpPublic privatePort publicPort type}}}}\"}"
#   -> take the {privatePort:22, isIpPublic:true} entry's ip + publicPort

# 3. First time on a fresh pod: install runtime deps (volume keeps code/models, not the venv).
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP> \
  "pip install -q --break-system-packages numpy pillow matplotlib"
```

## Prompt it

**Quick self-test (encode → decode → round-trip L1, saves a compare PNG):**
```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP> \
  "cd /workspace/sr && python -m eval.serve --ckpt checkpoints/tokenizer.pt --demo"
```

**HTTP service + SSH tunnel (prompt from your laptop, no extra ports exposed):**
```bash
# on the pod:
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP> \
  "cd /workspace/sr && python -m eval.serve --ckpt checkpoints/tokenizer.pt --http --port 8000"
# in another local terminal, tunnel 8000 and POST:
ssh -p <PORT> -i ~/.ssh/id_ed25519 -N -L 8000:localhost:8000 root@<IP>
curl localhost:8000/health
#   /encode    {"frames_b64": <b64 of np.save float32 (N,3,64,64) in [0,1]>} -> {"tokens": [[..256..]]}
#   /decode    {"tokens": [[..256..]]}                                       -> {"frames_b64": ..}
#   /roundtrip {"frames_b64": ..}                                            -> {"frames_b64":.., "l1":..}
```
(`eval/serve.py` has helper `_to_b64`/`_from_b64` for the numpy↔base64 wire format.)

## Tear down (stop GPU billing — the volume persists)
```bash
runpodctl pod delete <pod-id>
runpodctl pod list          # confirm []  (volume stays: runpodctl network-volume list)
```

Note: only the GPU pod bills per-hour; deleting it keeps everything on `sr-models`.
When M2 (dynamics) is trained, drop its checkpoint in `models/` and extend `serve.py`
with a `/dream` endpoint (encode frame + action → roll tokens → decode) — same volume,
same workflow.
```

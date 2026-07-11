"""Name -> builder registry for swappable tokenizers and dynamics cores.

Lets trainers/evals select a component by string (`build_dynamics("flow_bridge",
**cfg)`) instead of importing a concrete class, so a new variant is added by
registering it — no edits to the call sites. Defaults resolve to today's classes,
so nothing changes unless a different name is passed.

Registration is via decorator at factory-definition time; importing the module
that defines a component registers it. Built-in components are imported lazily on
the first build/list call (see `_ensure_registered`), so callers don't need to
know which modules to import first — and there are no __init__.py files to keep
in sync (these are namespace packages run via `python -m`).
"""

_TOKENIZERS = {}
_DYNAMICS = {}
_LOADED = False


def _ensure_registered():
    # Import the built-in component modules once so their @register_* decorators
    # run. Kept lazy + idempotent to avoid import cycles at module load.
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import model.tokenizer.fsq_autoencoder  # noqa: F401  registers "fsq"
    import model.tokenizer.vit_tokenizer     # noqa: F401  registers "fsq_vit" (stub)
    import model.dynamics.ar_core            # noqa: F401  registers "ar_transformer"
    import model.dynamics.flow_bridge        # noqa: F401  registers "flow_bridge"


def register_tokenizer(name):
    def deco(builder):
        if name in _TOKENIZERS:
            raise ValueError(f"tokenizer {name!r} already registered")
        _TOKENIZERS[name] = builder
        return builder
    return deco


def register_dynamics(name):
    def deco(builder):
        if name in _DYNAMICS:
            raise ValueError(f"dynamics {name!r} already registered")
        _DYNAMICS[name] = builder
        return builder
    return deco


def build_tokenizer(name, **cfg):
    _ensure_registered()
    return _build(_TOKENIZERS, "tokenizer", name, cfg)


def build_dynamics(name, **cfg):
    _ensure_registered()
    return _build(_DYNAMICS, "dynamics", name, cfg)


def tokenizer_names():
    _ensure_registered()
    return sorted(_TOKENIZERS)


def dynamics_names():
    _ensure_registered()
    return sorted(_DYNAMICS)


def _build(table, kind, name, cfg):
    if name not in table:
        raise KeyError(
            f"unknown {kind} {name!r}; registered: {sorted(table)}"
        )
    builder = table[name]
    # Trainers pass a superset cfg (e.g. the AR d_model/n_heads plus the bridge's
    # steps/hidden). Filter to the params THIS builder actually declares, so each
    # component ignores knobs meant for a different arch instead of erroring.
    return builder(**_accepted(builder, cfg))


def _accepted(builder, cfg):
    import inspect
    params = inspect.signature(builder).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cfg  # builder takes **kwargs; pass everything through
    return {k: v for k, v in cfg.items() if k in params}


# --- Checkpoints that carry their own construction config -------------------
# Trainers save {"builder": name, "cfg": {...}, "model": state_dict, ...}. The
# loaders rebuild via the registry so ANY variant (a bigger tokenizer after the
# sim upgrade, the flow bridge, a ViT) reloads exactly, with no hardcoded arch in
# the eval scripts. Old checkpoints (no builder key) fall back to the defaults.

def load_tokenizer(checkpoint, default_name="fsq", default_cfg=None, map_location="cpu"):
    return _load(build_tokenizer, checkpoint, default_name, default_cfg, map_location)


def load_dynamics(checkpoint, default_name="ar_transformer", default_cfg=None, map_location="cpu"):
    return _load(build_dynamics, checkpoint, default_name, default_cfg, map_location)


def _load(build_fn, checkpoint, default_name, default_cfg, map_location):
    import torch
    ckpt = torch.load(checkpoint, map_location=map_location)
    name = ckpt.get("builder", default_name)
    cfg = ckpt.get("cfg", default_cfg or {})
    model = build_fn(name, **cfg)
    model.load_state_dict(ckpt["model"])
    return model, ckpt

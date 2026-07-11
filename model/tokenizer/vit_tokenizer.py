"""ViT tokenizer variant (STUB — ablation slot, not yet implemented).

The default tokenizer uses a conv encoder (model/tokenizer/fsq_autoencoder.py).
This is the plug-and-play slot for the attention/ViT alternative Jai raised: a
patch-embed + transformer encoder that emits the SAME latent shape (B, G*G, C),
so the existing FSQ bottleneck and conv decoder are reused unchanged and only the
encoder differs. That makes conv-vs-ViT a clean ablation behind one `--tokenizer-
arch` flag.

Intended design when filled in:
  - Split the (B,3,64,64) frame into G*G patches (8x8 grid -> 8x8-pixel patches).
  - Linear patch embed -> add positional embeddings -> N transformer blocks.
  - Project each patch token to C channels -> (B, G*G, C) = the FSQ input.
  - Wrap with the shared FSQ + Decoder from fsq_autoencoder so decode is identical.

Left as a stub deliberately: the registry wiring is proven now, but the full
implementation waits until after the sim upgrade (bigger frames may change the
patch/grid sizing).
"""

from model.registry import register_tokenizer


@register_tokenizer("fsq_vit")
def _build_fsq_vit(**cfg):
    raise NotImplementedError(
        "fsq_vit tokenizer is a registered ablation stub — not implemented yet. "
        "See the module docstring for the intended patch-embed + transformer "
        "encoder design. Use the default 'fsq' tokenizer for now."
    )

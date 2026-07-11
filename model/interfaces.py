"""Structural interfaces for swappable model components.

Protocols (not base classes) so existing modules satisfy them without inheriting
anything — a class is a valid Tokenizer/Dynamics if it has the right methods.
This is the contract the registry, trainers, and evals depend on; concrete
implementations (FSQ, AR transformer, flow bridge, ViT, ...) plug in behind it.

Shape conventions (B = batch, tok = tokens_per_frame, V = visual codebook size):
  frame        (B, 3, H, W)  float in [0, 1]
  z_cont       (B, tok, C)   continuous pre-quant latent
  indices      (B, tok)      int64 visual code ids in [0, V)
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Compresses a frame to discrete visual tokens and back."""

    def encode(self, frame): ...
    def decode_indices(self, indices): ...   # (B, tok) -> (B, 3, H, W)
    def forward(self, frame): ...            # -> (recon, indices, z_cont)

    @property
    def codebook_size(self) -> int: ...      # number of distinct visual codes
    @property
    def tokens_per_frame(self) -> int: ...   # tokens emitted per frame


@runtime_checkable
class Dynamics(Protocol):
    """Predicts the next frame's visual tokens from past frames + actions."""

    # Training: one multi-step rollout loss over a batch. decoder maps predicted
    # visual tokens (B, tok) -> frame (B, 3, H, W) so pixel-space terms can be
    # computed without the dynamics core importing a tokenizer.
    def loss(self, batch, decoder): ...      # -> (total_loss, parts_dict)

    # Inference: generate one frame's visual tokens given the interleaved context
    # and the action for this step.
    def generate_frame(self, context_tokens, action_id): ...  # -> (B, tok)

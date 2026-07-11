- Option 1: Create physics engine, and allow an agent to invoke tools (not interesting)
- Web GPU, small model; need to focus on serving
- Find steering vectors for weather
- Model generates in real time
- Anchoring is the most important, need to avoid drift
- 3js driving sim which gives (frame, action, next frame)
    - Compression into latent representation
- First interpretable world model

1. Make driving sim (frame, action, next frame)
2. Build autoencoder
3. Build world model on top of autoencoder
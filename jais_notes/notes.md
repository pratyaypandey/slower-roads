# Notes
- Option 1: Create physics engine, and allow an agent to invoke tools (not interesting)
- Web GPU, small model; need to focus on serving
- Find steering vectors for weather
- Model generates in real time
- Anchoring is the most important, need to avoid drift
- 3js driving sim which gives (frame, action, next frame)
    - Compression into latent representation
- First interpretable world model
- This code will be trained on GPU but should be able to run on client side compute during inference
- Follow jais_notes/ai-guidelines.md

# Project Steps
1. Make driving sim (frame, action, next frame)
2. Build autoencoder
3. Build world model on top of autoencoder

# Paper Inspirations
- Dreamer v3
- Spatial tokenizer, for autoencoder maybe some kind of ViT
    - Paper: FINITE SCALAR QUANTIZATION: VQ-VAE MADE SIMPLE
    - Paper: MINEWORLD: A REAL-TIME AND OPEN-SOURCE INTERACTIVE WORLD MODEL ON MINECRAFT

# Tasks
0. Create a driving sim, similar to that of slowroads.io.
1. Create a mini training script for a world model, trained using FSQ, with the inference side inspired from Mineworld. 
2. After drafting the rough code for #1, think about different FSQ variants
    - I was thinking about how FSQ is a grid, but we should be able to do better than that maybe. I was thinking about AEP from information theory, and I was wondering if there are certain embeddings we think would be more likely (e.g. Gaussian or closer to the origin)
    - Experiment with different FSQ variants and see which one would be best for this project.
3. Create the pseudocode for visual tokenization and action tokenization, and then a network that takes that in and predicts the next frame, and then decodes. Make sure that multiple steps are simulated, decoded, and then compared for finding the loss. 
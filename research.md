# UniTok: A Unified Tokenizer for Visual Generation and Understanding

**Paper**: [arXiv:2502.20321](https://arxiv.org/abs/2502.20321)
**Project Page**: https://foundationvision.github.io/UniTok/
**Model**: [HuggingFace](https://huggingface.co/FoundationVision/unitok_tokenizer)
**Authors**: Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, Xiaojuan Qi
**Affiliations**: HKU, ByteDance, HUST
**Venue**: NeurIPS 2025 (Spotlight)

---

## 1. Motivation & Core Idea

Existing visual tokenizers are designed for either **generation** (VQ-GAN, RQ-VAE, VAR) or **understanding** (CLIP, SigLIP, ViTamin), but not both. UniTok bridges this gap by combining a VQ-VAE reconstruction pipeline with CLIP-style contrastive learning into a single model, producing discrete visual tokens that are simultaneously suitable for:

- **Autoregressive image generation** (e.g., LlamaGen)
- **Multimodal understanding** (e.g., LLaVA)
- **Unified multimodal large language models** (e.g., Chameleon, Liquid)

The key insight is that visual tokens can carry both fine-grained reconstruction information and high-level semantic alignment — if trained with the right combination of objectives.

---

## 2. Architecture Overview

```
Image (256×256)
    │
    ▼
┌──────────────────┐
│  ViTamin Encoder  │  (Hybrid CNN + ViT, patch_size=1)
│  MbConv → ViT     │  → (B, 256, 768)
└────────┬─────────┘
         │
    Linear Projection (768 → 64)
         │
         ▼
┌──────────────────────────────┐
│  Multi-Codebook VQ (×8)      │  8 codebooks, 4K entries each
│  Normalized embeddings       │  → (B, 8, 256) discrete indices
│  Entropy regularization      │  → (B, 256, 64) quantized features
└────────┬─────────────────────┘
         │
    Post-Quant Projection (64 → 768)
         │
         ├───────────────────────┐
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│ ViTamin Decoder  │    │ CLIP Projection   │
│ ViT → InvMbConv  │    │ MeanPool → FC     │
│ → (B, 3, 256, 256)   │ → (B, 768) norm   │
└─────────────────┘    └────────┬──────────┘
         │                       │
    Reconstructed Image    Contrastive Alignment
                           with Text Encoder
```

### 2.1 Encoder: ViTamin

- **Architecture**: Hybrid CNN-ViT from the ViTamin family (`vitamin_base` or `vitamin_large`)
- **Early stages**: MobileConv blocks (Stem → Stage1 → Stage2) with progressive downsampling
- **Late stage**: 14 ViT transformer blocks with GeGluMlp (gated linear unit MLP)
- **Config**: `patch_size=1`, no positional embedding, no class token, optional register tokens
- **Output**: `(B, 256, 768)` — 256 spatial tokens at 768 dimensions

### 2.2 Quantizer: Multi-Codebook VQ

- **Structure**: 8 independent codebooks, each with 4,096 entries (total effective vocab = 32,768)
- **Embedding dim**: 64 per token (split as 8 × 8 across codebooks)
- **Normalized embeddings**: Both input features and codebook entries are L2-normalized; nearest-neighbor lookup via `argmax(features @ codebook.T)`
- **VQ loss**: `β · MSE(quantized.detach(), features) + MSE(quantized, features.detach())` with `β=0.25`
- **Entropy loss** (optional): `L = E_sample - E_codebook`. Minimizing this encourages sharp per-sample assignments (low `E_sample`) and uniform codebook utilization (high `E_codebook`), preventing codebook collapse
- **Output codes**: `(B, 8, 16, 16)` — 8 codebook indices per spatial position

### 2.3 Decoder: Inverse ViTamin

- **Structure**: Symmetric to the encoder
- **Stage 1**: 14 ViT transformer blocks on quantized tokens
- **Stage 2–4**: Progressive upsampling via `InvMbConvLNBlock` with `Upsample2d` layers (16×16 → 32×32 → 128×128 → 256×256)
- **Output**: Reconstructed RGB image `(B, 3, 256, 256)` clamped to [-1, 1]

### 2.4 CLIP Text Tower

- 12-layer transformer, 768-dim, 12 heads
- BPE tokenizer with 49,408 vocab, context length 77
- Produces L2-normalized text features for contrastive learning
- Can be initialized from pretrained CLIP/ViTamin weights and optionally frozen

### 2.5 Discriminator: DINOv2-based

- **Backbone**: Frozen DINOv2 ViT-Small feature extractor
- **Heads**: 4 independent discriminator heads attached at DINOv2 layers {2, 5, 8, 11}
- **Each head**: 1×1 conv → Residual(9×9 conv) → SpectralNorm(1×1 conv → scalar)
- **Spectral normalization** on all conv layers for Lipschitz stability

---

## 3. Training

### 3.1 Dataset

- **DataComp-1B**: ~1.28 billion image-text pairs
- **Format**: WebDataset tar shards for distributed streaming
- **Image preprocessing**: Resize to 288 → center crop 256×256 → normalize to [-1, 1]
- **Text preprocessing**: BPE tokenization, padded to 77 tokens

### 3.2 Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| L2 reconstruction | 1.0 | Pixel-level fidelity |
| L1 reconstruction | 0.2 | Sharper details |
| LPIPS perceptual | 1.0 | Perceptual quality (VGG-based) |
| VQ commitment | 1.0 | Codebook alignment |
| Entropy | 0.0 (optional) | Prevent codebook collapse |
| CLIP contrastive | 1.0 | Semantic understanding |
| Discriminator (gen) | 0.4 | Adversarial realism |

**Total generator loss**:
```
L = (L1×0.2 + L2×1.0 + LPIPS×1.0) + VQ×1.0 + CLIP×1.0 + Disc×0.4
```

**Discriminator loss**: Hinge loss with Balanced Consistency Regularization (BCR) via DiffAug.

### 3.3 Training Schedule

| Phase | Timing | Description |
|-------|--------|-------------|
| Warmup | 0–1% of training | LR ramp from 0 to 5e-4 |
| Reconstruction + CLIP | 0–37.5% | Train encoder, decoder, quantizer, text tower |
| + Discriminator warmup | 37.5–40.5% | Gradually introduce adversarial loss |
| Full training | 40.5–100% | All losses active |

### 3.4 Optimization

- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.05)
- **Discriminator optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing with linear warmup
- **Precision**: BF16 mixed precision (AMP)
- **Distributed**: DDP across multiple nodes/GPUs

### 3.5 Adaptive Discriminator Weighting

The adversarial loss weight is dynamically adjusted based on gradient magnitudes:
```
w = ||∇L_reconstruction|| / ||∇L_adversarial||
```
This prevents the discriminator from dominating early in its warmup phase.

---

## 4. Inference Interface

```python
# Encode image to discrete tokens
code_idx = unitok.img_to_idx(img)        # (B, 8, 16, 16) indices

# Decode tokens back to image
rec_img = unitok.idx_to_img(code_idx)    # (B, 3, 256, 256) reconstructed

# Extract CLIP-aligned visual features (for understanding)
vis_feat = unitok.encode_image(img, normalize=True)   # (B, 768)

# Extract text features
txt_feat = unitok.encode_text(text, normalize=True)   # (B, 768)
```

---

## 5. Evaluation

### 5.1 Reconstruction Quality

| Tokenizer | rFID ↓ | Tokens |
|-----------|--------|--------|
| VQ-GAN | 4.98 | 256 |
| RQ-VAE | 1.30 | 256 |
| VAR | 0.90 | 680 |
| **UniTok** | **0.41** | **256** |

### 5.2 Zero-Shot Understanding

| Model | IN-1K Acc ↑ |
|-------|------------|
| CLIP | 76.2% |
| SigLIP | 80.5% |
| ViTamin | 81.2% |
| **UniTok** | **70.8%** |
| **UniTok (CLIP pretrain)** | **78.6%** |

### 5.3 Downstream: Unified MLLM (Liquid framework)

| Benchmark | VILA-U | UniTok |
|-----------|--------|--------|
| VQAv2 | 75.3 | **76.8** |
| GQA | 58.3 | **61.1** |
| TextVQA | 48.3 | **51.6** |
| MME | 1336 | **1448** |
| MM-Vet | 27.7 | **33.9** |

### 5.4 Downstream: Autoregressive Generation (LlamaGen)

UniTok notably favors generation **without classifier-free guidance** (CFG), reducing gFID from 14.6 (with CFG) to 2.51 (without CFG) — a significant finding for efficient sampling.

### 5.5 Evaluation Tooling

- **`utils/eval_fid.py`**: Computes FID and Inception Score via `torch_fidelity` on ImageNet val reconstructions
- **`utils/eval_acc.py`**: Zero-shot ImageNet classification using CLIP-style dot-product scoring
- **`eval/llamagen/`**: LlamaGen framework for class-conditional and text-conditional image generation evaluation
- **`eval/llava/`**: LLaVA two-stage training (alignment + instruction finetuning) for VQA benchmarks
- **`eval/liquid/`**: Liquid MLLM evaluation for both understanding and generation

---

## 6. Repository Structure

```
UniTok/
├── main.py                 # Training entry point (data, model, training loop, checkpointing)
├── trainer.py              # Trainer class (loss computation, discriminator scheduling, logging)
├── inference.py            # Image reconstruction demo
├── launch.sh               # Multi-node distributed training launcher
├── models/
│   ├── unitok.py           # Main UniTok model (encoder + quantizer + decoder + CLIP)
│   ├── vqvae.py            # Base VQVAE model (predecessor)
│   ├── quant.py            # VectorQuantizer and VectorQuantizerM
│   ├── vitamin.py          # ViTamin encoder/decoder architectures
│   ├── discrim.py          # DINOv2-based discriminator
│   ├── dinov2.py           # DINOv2 backbone
│   └── layers/             # Attention, MLP, patch embed, drop path, layer scale
├── utils/
│   ├── config.py           # All hyperparameters (Args dataclass)
│   ├── data.py             # WebDataset loading, CLIP transforms, text tokenization
│   ├── loss.py             # Discriminator loss variants (hinge, softplus, linear)
│   ├── lpips.py            # LPIPS perceptual loss
│   ├── optimizer.py        # LAMB, Lion, AmpOptimizer
│   ├── scheduler.py        # Cosine/linear/exponential LR schedules
│   ├── eval_acc.py         # Zero-shot ImageNet accuracy
│   ├── eval_fid.py         # FID and Inception Score
│   ├── dist.py             # Distributed training utilities
│   ├── diffaug.py          # Differentiable augmentation for discriminator
│   ├── visualizer.py       # Reconstruction visualization
│   └── logger.py           # Metrics logging (WandB integration)
├── open_clip/              # Modified OpenCLIP (text encoder, contrastive loss, tokenizer)
├── eval/
│   ├── EVAL.md             # Evaluation instructions
│   ├── liquid/             # Liquid MLLM evaluation
│   ├── llava/              # LLaVA training and evaluation
│   └── llamagen/           # LlamaGen generation evaluation
└── requirements.txt        # Dependencies (PyTorch ≥2.3.1, timm 1.0.8, webdataset, etc.)
```

---

## 7. Key Design Decisions

1. **Multi-codebook over single codebook**: 8 smaller codebooks (4K each) instead of one large codebook — enables better coverage with lower lookup cost and prevents codebook collapse.

2. **L2-normalized quantization**: Both encoder features and codebook embeddings are L2-normalized before nearest-neighbor lookup, stabilizing training and making the cosine-similarity-based lookup more robust.

3. **Progressive discriminator introduction**: The discriminator is inactive for the first 37.5% of training, allowing the VAE to converge before adversarial training begins. A 3% warmup further smooths the transition.

4. **ViTamin hybrid architecture**: Combines the efficiency of early CNN stages (MobileConv) with the expressiveness of ViT transformer blocks — better than pure ViT for reconstruction.

5. **GeGluMlp**: Gated Linear Unit MLP instead of standard MLP — provides better feature mixing through multiplicative gating.

6. **DINOv2 discriminator**: Using frozen self-supervised features (rather than learning from scratch) gives the discriminator strong multi-scale feature representations without training instability.

7. **Joint CLIP training**: Rather than post-hoc alignment, the CLIP contrastive loss is trained jointly, forcing the quantized tokens to preserve semantic structure.

8. **No CFG for generation**: UniTok's strong semantic tokens make classifier-free guidance unnecessary, simplifying and speeding up autoregressive sampling.

---

## 8. Dependencies & Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.3.1
- timm 1.0.8 (ViTamin models)
- webdataset 0.2.86 (distributed data loading)
- transformers (text encoder)
- kornia (differentiable image augmentation)
- torch_fidelity (FID/IS computation)
- numpy 1.26.4

---

## 9. Citation

```bibtex
@article{unitok,
  title={UniTok: A Unified Tokenizer for Visual Generation and Understanding},
  author={Ma, Chuofan and Jiang, Yi and Wu, Junfeng and Yang, Jihan and Yu, Xin and Yuan, Zehuan and Peng, Bingyue and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2502.20321},
  year={2025}
}
```

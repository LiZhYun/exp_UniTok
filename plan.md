# Percent-gated selective quantization for UniTok

## Core idea

Replace the standard VQ bottleneck with an error-aware gating mechanism that gradually transitions from continuous to fully discrete representations over the course of training. Tokens with high quantization error are bypassed (replaced with original features), controlled by a scheduled `percent` parameter that ramps from 0 to 1.

## Motivation

In standard VQ-VAE training, the codebook is randomly initialized and produces poor matches early on. The straight-through estimator forces the encoder to commit to these bad codes, causing slow convergence or codebook collapse. By gating out high-error tokens early and gradually increasing the quantization ratio, we create an implicit curriculum: easy tokens are quantized first, hard tokens get more time, and the encoder/decoder can establish good representations before the full discrete bottleneck is enforced.

## What to modify

All changes are scoped to two files:

1. `models/quant.py` — modify `VectorQuantizer.forward()`
2. `trainer.py` (or `main.py` training loop) — inject the percent schedule

No changes to the encoder, decoder, projection layers, or loss weighting logic.

## Implementation

### Step 1: Add percent buffer to `VectorQuantizer`

In `VectorQuantizer.__init__()`, add:

```python
self.register_buffer("percent", torch.tensor(1.0, dtype=torch.float32), persistent=False)
```

### Step 2: Modify `VectorQuantizer.forward()`

Replace the current forward method (lines 81–111 of `quant.py`). The new logic:

```python
def forward(self, features):
    B, L, C = features.shape
    features = features.reshape(-1, C)
    features = F.normalize(features, dim=-1).float()
    codebook_embed = self.codebook.get_norm_weight()

    # 1. find nearest code
    dists = features @ codebook_embed.T                    # (N, vocab_size)
    indices = torch.argmax(dists.detach(), dim=1)          # (N,)
    features_hat = self.codebook(indices)                  # (N, C)

    # 2. compute per-token quantization error
    quant_err = (features - features_hat).square().sum(dim=-1)  # (N,)

    # 3. straight-through estimator
    features_hat_ste = (features_hat.detach() - features.detach()).add_(features)

    # 4. percent-gated filtering (training only)
    if self.training and self.percent.item() < 1.0:
        # threshold: the p-th percentile of quant_err
        # tokens with error < threshold keep quantized value; others get original feature
        k = max(1, int(self.percent.item() * quant_err.numel()))
        thresh = quant_err.kthvalue(k).values
        mask = (quant_err <= thresh).unsqueeze(-1)         # (N, 1)
        features_gated = torch.where(mask, features_hat_ste, features)
    else:
        features_gated = features_hat_ste

    # 5. VQ loss
    #    codebook loss: pushes codebook toward ALL encoder features (not gated)
    #    commitment loss: uses gated output as target
    vq_loss = (
        F.mse_loss(features_hat, features.detach())
        + self.beta * F.mse_loss(features_gated.detach(), features)
    )

    # 6. entropy loss (computed on ALL tokens, not just gated ones)
    entropy_loss = (
        get_entropy_loss(features, codebook_embed, self.inv_entropy_tau)
        if self.use_entropy_loss else 0
    )

    # 7. vocab usage tracking (unchanged)
    prob_per_class_is_chosen = indices.bincount(minlength=self.vocab_size).float()
    handler = tdist.all_reduce(prob_per_class_is_chosen, async_op=True) if (
        self.training and dist.initialized()) else None
    if handler is not None:
        handler.wait()
    prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
    vocab_usage = (prob_per_class_is_chosen > 0.01 / self.vocab_size).float().mean().mul_(100)
    if self.vocab_usage_record_times == 0:
        self.vocab_usage.copy_(prob_per_class_is_chosen)
    elif self.vocab_usage_record_times < 100:
        self.vocab_usage.mul_(0.9).add_(prob_per_class_is_chosen, alpha=0.1)
    else:
        self.vocab_usage.mul_(0.99).add_(prob_per_class_is_chosen, alpha=0.01)
    self.vocab_usage_record_times += 1

    return features_gated.view(B, L, C), vq_loss, entropy_loss, vocab_usage
```

### Step 3: Propagate percent through `VectorQuantizerM`

In `VectorQuantizerM`, add a property so the percent buffer can be set from the training loop and applied to all sub-codebooks:

```python
def set_percent(self, value: float):
    for codebook in self.codebooks:
        codebook.percent.fill_(value)
```

### Step 4: Add the percent schedule to the training loop

In `main.py` or wherever the training loop lives, compute and inject the percent value each step:

```python
import numpy as np

def get_percent_schedule(step, total_steps, warmup_fraction=0.7, schedule="cosine"):
    """
    Returns a value in [0, 1].
    - For the first warmup_fraction of training, ramp from 0 to 1.
    - For the remaining steps, hold at 1.0 (full quantization).
    """
    warmup_steps = int(total_steps * warmup_fraction)
    if step >= warmup_steps:
        return 1.0
    t = step / warmup_steps
    if schedule == "cosine":
        return 0.5 * (1 - np.cos(np.pi * t))
    elif schedule == "linear":
        return t
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
```

In the training step, before `model(batch)`:

```python
percent = get_percent_schedule(global_step, total_steps)
unwrap_model(unitok).quantizer.set_percent(percent)
```

### Step 5: Logging

Log `percent` and the actual gated ratio per step for debugging:

```python
wandb_log({'quant_percent': percent}, step=global_step, log_ferq=50)
```

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Error metric | L2 distance in normalized space | Consistent with the cosine matching already used |
| Gating granularity | Per spatial token, all sub-codebooks together | Simpler; avoids mixed quantized/continuous within one token |
| Codebook loss scope | All tokens (not gated) | Codebook must learn from all features, not just easy ones |
| Entropy loss scope | All tokens (not gated) | Prevents codebook utilization collapse during low-percent phase |
| Commitment loss scope | Gated output | Bypassed tokens produce ~zero commitment loss by design |
| Schedule | Cosine 0→1 over 70% of training, hold 1.0 for last 30% | Long full-quant tail closes train/inference distribution gap |
| `f_to_idx` (inference) | Unchanged | Inference always uses full quantization, no gating |

## Hyperparameters to tune

- `warmup_fraction`: fraction of total steps for the 0→1 ramp (default: 0.7)
- `schedule`: "cosine" or "linear" (default: cosine)
- Optionally, `percent_start > 0` (e.g. 0.05) so the codebook receives some quantized-token signal from step 0

## Risks and mitigations

1. **Train/inference gap**: The decoder sees mixed continuous/discrete tokens during training. Mitigated by the long full-quantization tail (30% of training at `percent=1`).

2. **Codebook underutilization during low-percent phase**: Only easy tokens are quantized, which may concentrate on a few codes. Mitigated by computing entropy loss on all tokens' nearest-neighbor assignments.

3. **Gradient regime discontinuity at the threshold boundary**: Tokens near the threshold flip between two gradient paths across steps. In practice this should average out over batches, but monitor for loss spikes.

4. **Interaction with discriminator warmup**: UniTok already has a discriminator warmup schedule. Ensure the percent schedule and disc warmup don't create conflicting dynamics (e.g., don't ramp both simultaneously from zero).

## Validation plan

1. **Sanity check**: Run a short training with `percent=1.0` constant (equivalent to baseline). Verify identical behavior.
2. **Monitor codebook usage**: Plot vocab usage over training. It should not drop during the low-percent phase.
3. **Compare rFID and zero-shot accuracy** against the baseline at matched training steps.
4. **Ablate the schedule**: Compare cosine vs linear, and different `warmup_fraction` values (0.5, 0.7, 0.9).

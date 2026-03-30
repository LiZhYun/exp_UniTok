# Implementation Plan: Percent-Gated Selective Quantization

## Overview

Add a `--use_percent_gate` flag (default `False`) that enables percent-gated selective quantization. When disabled, behavior is identical to the original code. All changes are additive — no existing logic is modified.

---

## Files to Change

| File | What | Lines of new/changed code |
|------|------|---------------------------|
| `utils/config.py` | Add 3 new args | ~3 lines |
| `models/quant.py` | Modify `VectorQuantizer.forward()`, add `set_percent` to `VectorQuantizerM` | ~25 lines |
| `main.py` | Compute and inject percent schedule each step | ~15 lines |
| `trainer.py` | Log `quant_percent` to wandb | ~2 lines |

---

## Step 1: Add config flags

**File**: `utils/config.py`, inside `class Args`, after line 58 (`num_codebooks`).

```python
# percent-gated selective quantization
use_percent_gate: bool = False       # enable percent-gated VQ
percent_warmup_frac: float = 0.7     # fraction of total steps for 0→1 ramp
percent_schedule: str = 'cosine'     # 'cosine' or 'linear'
```

These go in the `# quantizer` section. When `use_percent_gate` is `False`, the entire feature is a no-op — original code path is taken.

---

## Step 2: Modify `VectorQuantizer`

**File**: `models/quant.py`

### 2a. Add percent buffer in `__init__` (after line 64)

```python
self.register_buffer("percent", torch.tensor(1.0, dtype=torch.float32), persistent=False)
```

This buffer defaults to 1.0 (= full quantization = original behavior). It is non-persistent so it won't pollute saved checkpoints.

### 2b. Replace `forward()` (lines 81–111)

The new forward method adds a gating branch guarded by `self.percent < 1.0 and self.training`. When `percent == 1.0` (the default), the code path is mathematically identical to the original.

```python
def forward(self, features):
    B, L, C = features.shape
    features = features.reshape(-1, C)
    features = F.normalize(features, dim=-1).float()
    codebook_embed = self.codebook.get_norm_weight()

    # nearest code lookup
    indices = torch.argmax(features.detach() @ codebook_embed.T, dim=1)
    entropy_loss = get_entropy_loss(features, codebook_embed, self.inv_entropy_tau) if self.use_entropy_loss else 0
    features_hat = self.codebook(indices)

    # straight-through estimator
    features_hat_ste = (features_hat.detach() - features.detach()).add_(features)

    # percent-gated filtering (training only, skip when percent == 1.0)
    if self.training and self.percent.item() < 1.0:
        quant_err = (features - features_hat).square().sum(dim=-1)       # (N,)
        k = max(1, int(self.percent.item() * quant_err.numel()))
        thresh = quant_err.kthvalue(k).values
        mask = (quant_err <= thresh).unsqueeze(-1)                        # (N, 1)
        features_gated = torch.where(mask, features_hat_ste, features)
    else:
        features_gated = features_hat_ste

    # VQ loss: codebook loss on all tokens, commitment loss uses gated output
    vq_loss = (
        F.mse_loss(features_hat, features.detach())
        + self.beta * F.mse_loss(features_gated.detach(), features)
    )

    # vocab usage tracking (unchanged from original)
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

**Key differences from the original `forward()`:**

| Aspect | Original (line ref) | New |
|--------|---------------------|-----|
| STE | Line 93: applied directly to `features_hat` | Stored in `features_hat_ste`, then optionally gated |
| Gating | None | When `percent < 1.0`: high-error tokens bypass quantization |
| Commitment loss target | `features_hat` (line 91) | `features_gated` — bypassed tokens produce ~zero commitment loss |
| Codebook loss | `features_hat` vs `features.detach()` | Same — codebook always learns from all tokens |
| When `percent == 1.0` | N/A | `features_gated = features_hat_ste`, numerically identical to original |

### 2c. Add `set_percent` to `VectorQuantizerM` (after line 147)

```python
def set_percent(self, value: float):
    for codebook in self.codebooks:
        codebook.percent.fill_(value)
```

---

## Step 3: Add percent schedule function and inject in training loop

**File**: `main.py`

### 3a. Add schedule function (top-level, after imports)

```python
def get_percent_schedule(step, total_steps, warmup_fraction=0.7, schedule='cosine'):
    """Returns a value in [0, 1]. Ramps from 0→1 over warmup_fraction of training, then holds at 1.0."""
    warmup_steps = int(total_steps * warmup_fraction)
    if step >= warmup_steps:
        return 1.0
    t = step / warmup_steps
    if schedule == 'cosine':
        return 0.5 * (1 - math.cos(math.pi * t))
    else:  # linear
        return t
```

Add `import math` at the top if not already present (it is not currently imported in `main.py`).

### 3b. Inject percent into training loop

**File**: `main.py`, inside `train_one_ep()`, right before `trainer.train_step(...)` call (line 123).

```python
# percent-gated schedule
if args.use_percent_gate:
    percent = get_percent_schedule(
        global_iter, args.epoch * num_iters,
        warmup_fraction=args.percent_warmup_frac,
        schedule=args.percent_schedule,
    )
    unwrap_model(trainer.unitok).quantizer.set_percent(percent)
```

Add `from utils.misc import unwrap_model` at the top of `main.py` (it already imports `misc`, so use `misc.unwrap_model` or add the direct import).

### 3c. Pass `percent` to `trainer.train_step`

Add `quant_percent` as an optional parameter to `train_step`:

```python
# in train_one_ep, at the trainer.train_step call:
trainer.train_step(
    ...
    quant_percent=percent if args.use_percent_gate else None,
)
```

---

## Step 4: Log percent in trainer

**File**: `trainer.py`

### 4a. Add `quant_percent` parameter to `train_step` signature (line 76)

```python
def train_step(
    self,
    img,
    text,
    global_iter: int,
    stepping: bool,
    metric_logger: MetricLogger,
    warmup_disc_schedule: float,
    fade_blur_schedule: float,
    report_wandb: bool = False,
    quant_percent: float = None,        # NEW
) -> ...:
```

### 4b. Add wandb logging (after line 245, in the `report_wandb` block)

```python
if quant_percent is not None:
    wandb_log({'Quant_percent': quant_percent}, step=global_iter, log_ferq=50)
```

---

## Summary of all edits

```
utils/config.py   +3 lines (new args)
models/quant.py   +1 line  (__init__ buffer)
                  ~10 lines (forward gating logic, replacing lines 81-111)
                  +3 lines  (set_percent method)
main.py           +1 line  (import math)
                  +14 lines (get_percent_schedule function)
                  +6 lines  (inject percent in train_one_ep)
trainer.py        +1 line  (new parameter)
                  +2 lines  (wandb log)
```

Total: ~40 lines of new code. Zero lines of original logic deleted — the original path is preserved behind the `use_percent_gate` flag.

---

## Verification checklist

1. **Baseline equivalence**: Run with `--use_percent_gate False` (default). The `percent` buffer stays at 1.0, the `if self.training and self.percent.item() < 1.0` branch is never taken, and `features_gated = features_hat_ste` is numerically identical to the original line 93.

2. **Sanity check**: Run with `--use_percent_gate True --percent_warmup_frac 0.0`. This sets `percent=1.0` from step 0, which should reproduce baseline exactly.

3. **Full test**: Run with `--use_percent_gate True` (default schedule). Monitor:
   - `Quant_percent` in wandb (should ramp 0→1 over 70% of training)
   - `Codebook_usage` (should not drop during low-percent phase)
   - `Lq` (VQ loss — should be lower early on due to gating)
   - rFID and zero-shot accuracy at convergence

4. **Ablations**: Compare `--percent_schedule cosine` vs `linear`, and `--percent_warmup_frac` values of 0.5, 0.7, 0.9.

# Training

This directory contains training scripts for ATOMIC model variants.

---

## ATOMIC-LLaVA

ATOMIC-LLaVA is trained using the [LLaVA](https://github.com/haotian-liu/LLaVA) training framework without modification.

Please follow LLaVA's installation and training guide, and use our Stage 1 and Stage 2 training data from HuggingFace:

👉 [https://huggingface.co/datasets/LabSmart/ATOMIC_dataset](https://huggingface.co/datasets/LabSmart/ATOMIC_dataset)

Training data format is compatible with LLaVA's default data format.

---

## ATOMIC-Gemma

ATOMIC-Gemma is trained from `google/gemma-3-4b-it` using Stage 2 data only.

> **Note:** ATOMIC-Gemma is developed after the ECCV 2026 submission deadline and is **not part of the published paper**. It is provided here to demonstrate the generalizability of the ATOMIC training pipeline across different base model architectures.

### Files

| File | Description |
|------|-------------|
| `atomic_gemma/convert_to_gemma_format.py` | Convert LLaVA JSON format to Gemma3 Vision format |
| `atomic_gemma/train_gemma3.py` | SFT training script for Gemma3-4B-IT |

### Installation

```bash
# Install remaining dependencies
pip install -r atomic_gemma/requirements.txt
```

### Usage

**Step 1 — Convert Stage 2 data to Gemma format**

```bash
python atomic_gemma/convert_to_gemma_format.py \
  --input /path/to/your/Stage2_blend_60k.json \
  --output /path/to/your/Conversion.json
```

**Step 2 — Run training**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=4 atomic_gemma/train_gemma3.py
```

To run in test mode (10 samples only):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=4 atomic_gemma/train_gemma3.py --test
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-3-4b-it` |
| Training stage | Stage 2 only |
| Frozen | Vision encoder (`vision_tower`) |
| Trainable | Connector (`multi_modal_projector`) + LLM (`language_model`) |
| Batch size | 16 per device |
| Learning rate | 2e-5 |
| LR scheduler | Cosine |
| Precision | bfloat16 |
| Epochs | 1 |
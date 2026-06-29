# Evaluation

This directory contains evaluation scripts for three benchmarks: **TEM-VQA**, **TEM-MCQ**, and **MatCha**.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note for ATOMIC-LLaVA inference:** LLaVA-based inference scripts (`inference_*_llava.py`) must be run from within the [LLaVA repository](https://github.com/haotian-liu/LLaVA). Please follow LLaVA's installation guide first, then run the scripts from that environment.

---

## Directory Structure

```
evaluation/
├── TEM_VQA/
│   ├── utils_tem_vqa.py
│   ├── inference_TEM_VQA_llava.py
│   ├── inference_TEM_VQA_gemma.py
│   ├── run_TEM_VQA_parallel.py
│   └── score.py
├── TEM_MCQ/
│   ├── utils_tem_mcq.py
│   ├── inference_TEM_MCQ_llava.py
│   ├── inference_TEM_MCQ_gemma.py
│   ├── run_TEM_MCQ_parallel.py
│   └── score_single.py
└── matcha/
    ├── utils_matcha.py
    ├── inference_matcha_llava.py
    ├── inference_matcha_gemma.py
    ├── run_matcha_parallel.py
    ├── score_matcha.py
    └── manual_labels.jsonl
```

---

## TEM-VQA

TEM-VQA is an open-ended short-answer benchmark evaluated using AWC, BLEU-1, ROUGE-L, and METEOR.

**Step 1 — Run inference (multi-GPU parallel):**
```bash
python TEM_VQA/run_TEM_VQA_parallel.py \
    --inference_script TEM_VQA/inference_TEM_VQA_llava.py \
    --input_csv /path/to/your/TEM_VQA.csv \
    --image_dir /path/to/your/TEM_figures \
    --output_dir /path/to/your/outputs \
    --num_gpus 4
```

**Step 2 — Compute scores:**
```bash
python TEM_VQA/score.py --output_dir /path/to/your/outputs
```

---

## TEM-MCQ

TEM-MCQ is a multiple-choice benchmark evaluated by per-question accuracy across three categories: Visual Perception, Scientific Reasoning, and Experimental Design.

**Step 1 — Run inference (multi-GPU parallel):**
```bash
python TEM_MCQ/run_TEM_MCQ_parallel.py \
    --inference_script TEM_MCQ/inference_TEM_MCQ_llava.py \
    --input_csv /path/to/your/TEM_MCQ.csv \
    --image_dir /path/to/your/TEM_figures \
    --output_dir /path/to/your/outputs \
    --num_gpus 4
```

**Step 2 — Compute scores:**
```bash
python TEM_MCQ/score.py --output_dir /path/to/your/outputs
```

---

## MatCha

MatCha is an external materials science VQA benchmark. To obtain the MatCha dataset, please refer to the original repository:

👉 [https://github.com/FreedomIntelligence/MatCha](https://github.com/FreedomIntelligence/MatCha)

We provide `matcha/manual_labels.jsonl`, which contains our manually annotated TEM image subset from the MatCha dataset. This file is required for computing scores on the TEM-only subset used in our paper.

**Step 1 — Run inference (multi-GPU parallel):**
```bash
python matcha/run_matcha_parallel.py \
    --inference_script matcha/inference_matcha_llava.py \
    --input_jsonl /path/to/your/matcha_vqa_tem_only.jsonl \
    --image_dir /path/to/your/matcha_images \
    --output_dir /path/to/your/outputs \
    --num_gpus 4
```

**Step 2 — Compute scores:**
```bash
python matcha/score.py \
    --output_dir /path/to/your/outputs \
    --input_jsonl /path/to/your/matcha_vqa_inputs.jsonl \
    --label_jsonl matcha/manual_labels.jsonl
```

---

## Switching Models

Each inference script supports both the baseline and ATOMIC variants. Edit `MODEL_PATH` at the top of the script:

```python
# Baseline
# MODEL_PATH = "liuhaotian/llava-v1.5-7b"      # LLaVA
# MODEL_PATH = "google/gemma-3-4b-it"           # Gemma3

# ATOMIC
MODEL_PATH = "/path/to/your/ATOMIC-llava"
# MODEL_PATH = "/path/to/your/ATOMIC-gemma"
```
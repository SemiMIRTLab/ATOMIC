# Instruction Generation

This directory contains scripts for generating ATOMIC training and evaluation data using GPT-4.1.

---

## Prerequisites: Prepare `total_dataset.csv`
 
Before running any notebook in this directory, you need to prepare `total_dataset.csv`, which is the starting point for the entire instruction generation pipeline.
 
This file is produced by merging two sources:
 
- **Crawling output** (`dataset_nature.csv`, `dataset_ncomms.csv`, etc.) — provides `ARTICLE_ID`, `FIG_ID`, `CAPTION`, `ARTICLE_URL`
- **Subfigure extraction output** (`results.csv`) — provides `CROP_IMAGE`, `RESNET_PRED_CLASS`
The merged `total_dataset.csv` should contain the following columns:
 
| Column | Description |
|--------|-------------|
| `IMAGE_PATH` | Original compound figure filename |
| `CROP_IMAGE` | Extracted subfigure filename |
| `RESNET_PRED_CLASS` | TEM modality: `CTEM`, `HR-TEM`, `STEM`, `Diffraction` |
| `ARTICLE_ID` | Source article identifier |
| `FIG_ID` | Figure identifier within the article |
| `TITLE` | Figure title |
| `CAPTION` | Parent figure caption |
| `ARTICLE_URL` | URL to the source article |
| `OPEN_ACCESS` | Whether the article is open access |

---

## Execution Order

Run the notebooks in the following order:

**1. `dataset_split.ipynb`**
Splits `total_dataset.csv` into `train.csv` and `test.csv`, ensuring no `ARTICLE_ID` overlap between splits.

**2. `stage1_generate_descriptions.ipynb`**
For each subfigure in `train.csv`, generates VisionGround and DomainContext descriptions using GPT-4.1.

**3. `stage2_generate_conversations.ipynb`**
Converts Stage 1 descriptions into multi-turn VisionGround and DomainContext conversations for Stage 2 training.

**4. `generate_MCQ.ipynb`**
For each subfigure in `test.csv`, generates 9 multiple-choice questions across three categories (Visual Perception, Scientific Reasoning, Experimental Design) and three difficulty levels. Half of the generated MCQ data is further converted into TEM-VQA (short open-ended answer format).

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Environment Variables

All notebooks read the OpenAI API key from the environment:

```bash
export OPENAI_API_KEY=your_key_here
```
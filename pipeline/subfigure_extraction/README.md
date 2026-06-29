# Subfigure Extraction

This directory contains the script for extracting TEM subfigures from compound figures downloaded from academic articles.

> **Note:** If your goal is to obtain the subfigures used in our paper, you do **not** need to run this script. Please refer to `data/README.md` for the faster reconstruction path using `TEM_source_index.csv`.
>
> This script is intended for users who wish to **build their own dataset from scratch** using newly crawled figures.

---

## Pipeline Overview

`subfigure_extraction.ipynb` runs a two-stage extraction pipeline:

1. **YOLO** — detects and crops individual subfigures from each compound figure
2. **ResNet** — classifies each crop into one of four TEM modalities: `CTEM`, `HR-TEM`, `STEM`, `Diffraction`

Only crops classified as one of the target modalities are saved.

---

## Model Weights

YOLO and ResNet weights are **not released** in this repository as they are specific to our preprocessing pipeline. If you wish to retrain:

- **YOLO**: train on compound TEM figures with subfigure bounding box annotations
- **ResNet**: train a `resnet50` classifier on TEM crops with modality labels (`CTEM`, `HR-TEM`, `STEM`, `Diffraction`), input size `384 x 384`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

1. Place your downloaded compound figures in your `raw_figures` directory
2. Set the paths in the Settings cell of the notebook
3. Run all cells

```bash
jupyter notebook subfigure_extraction.ipynb
```

---

## Output

| Output | Description |
|--------|-------------|
| `TEM_figures/` | Extracted and classified TEM subfigures (`.png`) |
| `results.csv` | Record of each subfigure and its predicted modality |

`results.csv` columns:

| Column | Description |
|--------|-------------|
| `source_image` | Original compound figure filename |
| `sub_image` | Extracted subfigure filename (`{base}_crop{N}.png`) |
| `resnet_pred_class` | Predicted TEM modality |
# Data

This directory contains the dataset resources for ATOMIC.

---

## Quick Start: Reconstruct Subfigures from Source Index

If you want to obtain the TEM subfigures used in our paper, follow these two steps:

### Step 1 — Download original figures

Each row in `TEM_source_index.csv` contains an `ARTICLE_URL` and a `FIG_ID` that together point to the original figure page. The figure page URL can be constructed as:

```
{ARTICLE_URL}/figures/{N}
```

where `{N}` is the figure number extracted from `FIG_ID` (e.g., `figure-3` → `3`).

For example:
- `ARTICLE_URL`: `https://www.nature.com/articles/nature08692`
- `FIG_ID`: `figure-3`
- Figure page: `https://www.nature.com/articles/nature08692/figures/3`

Download the figure image from the figure page and save it as:

```
{ARTICLE_ID}_{FIG_ID}.jpg
```

For example, `nature08692_figure-3.jpg`.

> **Note:** Access to some articles requires institutional subscription. Only figures from open-access articles (`OPEN_ACCESS: True`) can be downloaded freely.

### Step 2 — Crop subfigures

Run `reconstruct_subfigures.ipynb`, which reads `TEM_source_index.csv` and uses the normalized crop coordinates (`X_CENTER`, `Y_CENTER`, `WIDTH`, `HEIGHT`) to crop each subfigure from the downloaded figures.

```bash
jupyter notebook reconstruct_subfigures.ipynb
```

Output subfigures will be saved to `Data/TEM_subfigures/`, named by `TEM_SUB_IMAGE_ID`.

---

## Full Pipeline (Build Your Own Dataset from Scratch)

If you want to collect your own TEM dataset rather than reconstructing ours, refer to `pipeline/` instead:

```
pipeline/crawling/             # Crawl articles from Nature portfolio journals
pipeline/subfigure_extraction/ # Extract and classify TEM subfigures (YOLO + ResNet)
pipeline/instruction_generation/ # Generate training data with GPT
```

---

## TEM_source_index.csv

The source index contains metadata for all 32,564 TEM subfigures in our dataset.

| Column | Description |
|--------|-------------|
| `TEM_SUB_IMAGE_ID` | Subfigure filename (`{ARTICLE_ID}_{FIG_ID}_crop{N}.png`) |
| `RESNET_PRED_CLASS` | TEM modality: `CTEM`, `HR-TEM`, `STEM`, `Diffraction` |
| `X_CENTER` | Normalized x-center of crop bounding box |
| `Y_CENTER` | Normalized y-center of crop bounding box |
| `WIDTH` | Normalized width of crop bounding box |
| `HEIGHT` | Normalized height of crop bounding box |
| `ARTICLE_ID` | Source article identifier |
| `FIG_ID` | Figure identifier within the article |
| `TITLE` | Figure title |
| `CAPTION` | Figure caption |
| `ARTICLE_URL` | URL to the source article |
| `OPEN_ACCESS` | Whether the article is open access |

> **Copyright Notice:** The subfigure images themselves are not distributed in this repository due to copyright restrictions. `TEM_source_index.csv` provides reconstruction metadata only. Users are responsible for complying with the terms of use of the respective publishers when accessing source articles.
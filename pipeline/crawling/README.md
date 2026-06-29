# Crawling

This directory contains scripts for crawling TEM-related research articles from Nature portfolio journals.

> **Note:** If your goal is to reconstruct the dataset used in our paper, you do **not** need to run these scripts. Please refer to `data/TEM_source_index.csv` instead, which provides article URLs, figure IDs, and normalized crop coordinates for sub-figure reconstruction.
>
> These scripts are intended for users who wish to **build their own dataset from scratch** — for example, targeting different journals, subjects, or date ranges.

---

## Journals Covered

| Script | Journal | Journal Code |
|--------|---------|--------------|
| `Nature_crawler.py` | Nature | `nature` |
| `Nature_Communications_crawler.py` | Nature Communications | `ncomms` |
| `Nature_Materials_crawler.py` | Nature Materials | `nmat` |
| `Nature_Nanotechnology_crawler.py` | Nature Nanotechnology | `nnano` |

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python Nature_crawler.py --year 2010 --max-articles 1000 --output-dir ./downloads_nature
python Nature_Communications_crawler.py --year 2010 --max-articles 1000 --output-dir ./downloads_ncomms
python Nature_Materials_crawler.py --year 2010 --max-articles 1000 --output-dir ./downloads_nmat
python Nature_Nanotechnology_crawler.py --year 2010 --max-articles 1000 --output-dir ./downloads_nnano
```



> **Note:** Each script writes to its own CSV file (`dataset_nature.csv`, `dataset_ncomms.csv`, `dataset_nmat.csv`, `dataset_nnano.csv`) and a separate output directory by default, so running multiple crawlers simultaneously will not cause conflicts.

---

## Output

Each script produces:
- `downloads_<journal>/` — Downloaded figure images (`{article_id}_{fig_id}.jpg`)
- `dataset_<journal>.csv` — Figure metadata with the following columns:

| Column | Description |
|--------|-------------|
| `ARTICLE_ID` | Unique article identifier |
| `FIG_ID` | Figure identifier within the article |
| `TITLE` | Figure title |
| `CAPTION` | Figure caption (citation superscripts removed) |
| `IMAGE_PATH` | Local path to downloaded figure image |
| `ARTICLE_URL` | Original article URL |
| `OPEN_ACCESS` | Whether the article is open access |

> **Note:** Downloaded images are not included in this repository due to copyright restrictions. The `TEM_source_index.csv` in `data/` is derived from this output and provides image reconstruction metadata for researchers with lawful access to the source articles.
 
---
 
## Legal Notice
 
These scripts are provided for academic research transparency only. Users are responsible for complying with the terms of service of the respective publishers before running them.
"""
Compute TEM-VQA scores (BLEU-1, ROUGE-L, METEOR, AWC) from inference outputs.

Usage:
    python score.py --output_dir /path/to/your/outputs

Expects files matching: TEM_VQA_*_outputs.csv
"""

import pandas as pd
import os
import glob
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

for _resource in ["wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory containing TEM_VQA_*_outputs.csv files')
args = parser.parse_args()

INPUT_FILES     = glob.glob(os.path.join(args.output_dir, "TEM_VQA_*_outputs.csv"))
DIFFICULTY_CYCLE = ["easy", "medium", "hard"]
TASK_TYPE_ORDER  = ["visual_perception", "reasoning", "experimental_design"]

rouge_sc  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothing = SmoothingFunction().method1


def model_label(filename: str) -> str:
    return filename.replace("TEM_VQA_", "").replace("_outputs.csv", "")


def normalize(text):
    import re
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())


def compute_scores(ref: str, hyp: str) -> dict:
    rt = ref.split() or [""]
    ht = hyp.split() or [""]
    bleu1  = sentence_bleu([rt], ht, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    rougeL = rouge_sc.score(ref, hyp)["rougeL"].fmeasure
    meteor = meteor_score([rt], ht) if (rt and ht) else 0.0
    if not rt:
        awc = 0.0
    else:
        hyp_cnt = {}
        for w in ht:
            hyp_cnt[w] = hyp_cnt.get(w, 0) + 1
        ref_cnt = {}
        for w in rt:
            ref_cnt[w] = ref_cnt.get(w, 0) + 1
        matched = sum(min(c, hyp_cnt.get(w, 0)) for w, c in ref_cnt.items())
        awc = matched / len(rt)
    return {"BLEU-1": bleu1, "ROUGE-L": rougeL, "METEOR": meteor, "AWC": awc}


def mean_scores(rows: list) -> dict:
    n = len(rows)
    if n == 0:
        return {k: None for k in ["BLEU-1", "ROUGE-L", "METEOR", "AWC"]}
    return {k: round(sum(r[k] for r in rows) / n * 100, 1) for k in ["BLEU-1", "ROUGE-L", "METEOR", "AWC"]}


def fmt(v):
    return f"{v:.1f}" if v is not None else "N/A"


def print_table(title, rows, cols):
    col_widths = [max(len(c), 10) + 2 for c in cols]
    col_widths[0] = 40
    header = "".join(
        f"{c:<{col_widths[i]}}" if i == 0 else f"{c:>{col_widths[i]}}"
        for i, c in enumerate(cols)
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n  {title}\n{sep}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: x.get(cols[-1]) or -1, reverse=True):
        line = "".join(
            f"{str(r.get(c, 'N/A')):<{col_widths[i]}}" if i == 0
            else f"{fmt(r.get(c)):>{col_widths[i]}}"
            for i, c in enumerate(cols)
        )
        print(line)
    print(sep)


files = [f for f in INPUT_FILES if os.path.exists(f)]
if not files:
    print(f"No output files found in: {args.output_dir}")
    exit(1)

print(f"Found {len(files)} file(s), computing scores...\n")

overall_rows = []
task_rows    = []

for fpath in files:
    df = pd.read_csv(fpath, dtype=str).fillna("")
    df.columns = [c.upper() for c in df.columns]
    if "RESPONSE" not in df.columns or "CORRECT_ANSWER" not in df.columns:
        print(f"[SKIP] {fpath} missing required columns")
        continue

    df["DIFFICULTY"] = [DIFFICULTY_CYCLE[i % 3] for i in range(len(df))]
    model = model_label(os.path.basename(fpath))

    scores = []
    for _, row in df.iterrows():
        hyp = normalize(row["RESPONSE"])
        ref = normalize(row["CORRECT_ANSWER"])
        scores.append(compute_scores(ref, hyp))

    overall_rows.append({"Model": model, **mean_scores(scores)})

    task_row = {"Model": model}
    if "TASK_TYPE" in df.columns:
        for ttype in TASK_TYPE_ORDER:
            idx = df.index[df["TASK_TYPE"].str.lower() == ttype.lower()].tolist()
            sub = [scores[i] for i in idx]
            task_row[ttype] = mean_scores(sub)["AWC"] if sub else None
    task_rows.append(task_row)

print_table(
    "Table 1 — Overall Results  (BLEU-1 / ROUGE-L / METEOR / AWC)",
    overall_rows,
    ["Model", "BLEU-1", "ROUGE-L", "METEOR", "AWC"]
)
print_table(
    "Table 2 — AWC by Task Type",
    task_rows,
    ["Model"] + TASK_TYPE_ORDER
)
"""
Compute TEM-MCQ accuracy from inference outputs.

Usage:
    python score.py --output_dir /path/to/your/outputs

Expects files matching: TEM_MCQ_*_outputs.csv
"""

import pandas as pd
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory containing TEM_MCQ_*_outputs.csv files')
args = parser.parse_args()

INPUT_FILES      = glob.glob(os.path.join(args.output_dir, "TEM_MCQ_*_outputs.csv"))
DIFFICULTY_CYCLE = ["easy", "medium", "hard"]
TASK_TYPE_ORDER  = ["visual_perception", "reasoning", "experimental_design"]


def model_label(filename: str) -> str:
    return filename.replace("TEM_MCQ_", "").replace("_outputs.csv", "")


def per_question_accuracy(sub_df):
    if sub_df.empty:
        return None
    correct = sub_df["_correct"].sum()
    total   = len(sub_df)
    return round(correct / total * 100, 1)


files = [f for f in INPUT_FILES if os.path.exists(f)]
if not files:
    print(f"No output files found in: {args.output_dir}")
    exit(1)

print(f"Found {len(files)} file(s), computing scores...\n")

rows = []
for fpath in files:
    df = pd.read_csv(fpath, dtype=str).fillna("")
    df.columns = [c.upper() for c in df.columns]
    if "RESPONSE" not in df.columns or "CORRECT_ANSWER" not in df.columns:
        print(f"[SKIP] {fpath} missing required columns")
        continue

    df["DIFFICULTY"] = [DIFFICULTY_CYCLE[i % 3] for i in range(len(df))]
    df["_correct"] = (
        (df["RESPONSE"] == df["CORRECT_ANSWER"].str.strip().str.upper()) &
        (df["RESPONSE"] != "")
    )

    model = model_label(os.path.basename(fpath))
    row   = {"Model": model}

    for ttype in TASK_TYPE_ORDER:
        sub = df[df["TASK_TYPE"].str.lower() == ttype.lower()]
        row[ttype] = per_question_accuracy(sub)

    row["Overall"] = per_question_accuracy(df)
    row["n"]       = len(df)
    rows.append(row)


def fmt(v):
    return f"{v:.1f}" if v is not None else "N/A"


def print_table(title, df):
    header = (
        f"{'Model':<40} "
        f"{'visual_perception':>18} "
        f"{'reasoning':>12} "
        f"{'experimental_design':>20} "
        f"{'Overall':>10} "
        f"{'n':>6}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n  {title}\n{sep}")
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        print(
            f"{r['Model']:<40} "
            f"{fmt(r['visual_perception']):>18} "
            f"{fmt(r['reasoning']):>12} "
            f"{fmt(r['experimental_design']):>20} "
            f"{fmt(r['Overall']):>10} "
            f"{int(r['n']):>6}"
        )
    print(sep)


cols   = ["Model"] + TASK_TYPE_ORDER + ["Overall", "n"]
df_out = (
    pd.DataFrame(rows)[cols]
    .sort_values("Overall", ascending=False)
    .reset_index(drop=True)
)

print_table("Per-Question Accuracy (%)", df_out)
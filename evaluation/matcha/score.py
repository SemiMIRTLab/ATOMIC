"""
Compute MatCha benchmark scores from inference outputs.

Usage:
    python score.py \
        --output_dir /path/to/your/outputs \
        --input_jsonl /path/to/your/matcha_vqa_inputs.jsonl \
        --label_jsonl /path/to/your/manual_labels.jsonl
"""

import json
import glob
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',  type=str, required=True,
                    help='Directory containing matcha_*_outputs.jsonl files')
parser.add_argument('--input_jsonl', type=str, required=True,
                    help='Path to matcha_vqa_inputs.jsonl')
parser.add_argument('--label_jsonl', type=str, required=True,
                    help='Path to manual_labels.jsonl (TEM label file)')
args = parser.parse_args()

TOPIC_CATEGORIES = {
    "Morphology Analysis": [
        "Material Classification",
        "Image Content Analysis",
        "Surface Microstructure Assessment",
        "Surface Roughness Assessment",
        "Defect Type Classification",
        "Grain/Pore Size Classification",
    ],
    "Structure Analysis": [
        "Crystallographic Data Inference",
        "Crystallinity Classification",
        "Multiphase Interface Assessment",
        "XRD Pattern Analysis",
        "Phase Analysis",
        "Elemental Mapping Analysis",
        "Element Distribution Homogeneity Assessment",
        "Material Morphology and Composition Uniformity Assessment",
    ],
    "Property Analysis": [
        "Physical and Chemical Properties Inference",
        "Mechanical Properties Analysis",
        "Thermal Analysis",
        "Infrared (IR) and Raman (RS) Spectral Analysis",
        "XPS Spectrum Analysis",
    ],
    "Processing Correlation": [
        "Characterization Technique Identification",
        "Characterization Purpose Inference",
    ],
}

topic_to_category = {}
for cat, topics in TOPIC_CATEGORIES.items():
    for t in topics:
        topic_to_category[t] = cat

CAT_SHORT = {
    "Processing Correlation": "PC",
    "Morphology Analysis":    "MA",
    "Structure Analysis":     "SA",
    "Property Analysis":      "PA",
}

DISPLAY_COLS = ["PC", "MA", "SA", "PA", "Suppl. DTC", "ALL"]


def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_tem_ids(label_path):
    tem_ids = set()
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["label"] == "tem":
                tem_ids.add(r["id"])
    return tem_ids

def calc_score(inputs_map, outputs_path, tem_ids):
    outputs = load_dataset(outputs_path)
    outputs_map = {o["id"]: o for o in outputs}

    cat_correct  = defaultdict(int)
    cat_total    = defaultdict(int)
    suppl_correct, suppl_total = 0, 0

    for record_id, sample in inputs_map.items():
        sid    = record_id.split("-", 2)[-1]
        output = outputs_map.get(record_id)
        if output is None:
            continue

        if sid.startswith("IDMLAMCS") or sid.startswith("UHCSDB"):
            continue

        if sid.startswith("LDRQSMA"):
            suppl_correct += sum(output["correct"])
            suppl_total   += len(output["correct"])
            continue

        if record_id not in tem_ids:
            continue

        for j, vqa in enumerate(sample.get("vqa", [])):
            topic = vqa.get("topic")
            cat   = topic_to_category.get(topic)
            if cat is None:
                continue
            short = CAT_SHORT[cat]
            cat_correct[short] += int(output["correct"][j])
            cat_total[short]   += 1

    return cat_correct, cat_total, suppl_correct, suppl_total


def print_table(title, all_rows, col_totals):
    pc_n  = col_totals.get("PC", 0)
    ma_n  = col_totals.get("MA", 0)
    sa_n  = col_totals.get("SA", 0)
    pa_n  = col_totals.get("PA", 0)
    sup_n = col_totals.get("Suppl. DTC", 0)
    all_n = pc_n + ma_n + sa_n + pa_n + sup_n
    header = (
        f"{'Model':<45} "
        f"{f'PC (n={pc_n})':>14} "
        f"{f'MA (n={ma_n})':>14} "
        f"{f'SA (n={sa_n})':>14} "
        f"{f'PA (n={pa_n})':>14} "
        f"{f'Suppl. DTC (n={sup_n})':>22} "
        f"{f'ALL (n={all_n})':>16}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n  {title}\n{sep}")
    print(header)
    print("-" * len(header))
    for r in sorted(all_rows, key=lambda x: x["_ALL_val"] if x["_ALL_val"] is not None else -1, reverse=True):
        print(
            f"{r['Model']:<45} "
            f"{r['PC']:>14} "
            f"{r['MA']:>14} "
            f"{r['SA']:>14} "
            f"{r['PA']:>14} "
            f"{r['Suppl. DTC']:>22} "
            f"{r['ALL']:>16}"
        )
    print(sep)


inputs     = load_dataset(args.input_jsonl)
tem_ids    = load_tem_ids(args.label_jsonl)
inputs_map = {inp["id"]: inp for inp in inputs}

output_files = glob.glob(os.path.join(args.output_dir, "matcha_*_outputs.jsonl"))
if not output_files:
    print(f"No output files found in: {args.output_dir}")
    exit(1)

print(f"Found {len(output_files)} file(s), computing scores...\n")

all_rows = []
for fpath in output_files:
    model = os.path.basename(fpath).replace("matcha_", "").replace("_outputs.jsonl", "")
    cat_correct, cat_total, suppl_correct, suppl_total = calc_score(inputs_map, fpath, tem_ids)

    row = {"Model": model}
    all_c, all_t = 0, 0

    for short in ["PC", "MA", "SA", "PA"]:
        c, t = cat_correct[short], cat_total[short]
        row[short] = f"{c/t*100:.1f}%" if t > 0 else "N/A"
        all_c += c
        all_t += t

    row["Suppl. DTC"] = f"{suppl_correct/suppl_total*100:.1f}%" if suppl_total > 0 else "N/A"
    all_c += suppl_correct
    all_t += suppl_total

    row["ALL"]      = f"{all_c/all_t*100:.1f}%" if all_t > 0 else "N/A"
    row["_ALL_val"] = all_c / all_t * 100 if all_t > 0 else None
    all_rows.append(row)

# Compute column totals from first file
col_totals = {}
if output_files:
    cat_c, cat_t, sc, st = calc_score(inputs_map, output_files[0], tem_ids)
    for short in ["PC", "MA", "SA", "PA"]:
        col_totals[short] = cat_t[short]
    col_totals["Suppl. DTC"] = st

print_table("Ranking by ALL  (TEM only, per VQA, by topic category)", all_rows, col_totals)
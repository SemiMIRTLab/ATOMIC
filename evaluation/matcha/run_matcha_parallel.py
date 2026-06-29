"""
Run MatCha benchmark inference in parallel across multiple GPUs.

Each GPU processes a separate chunk of the input JSONL.
Results are merged into a single output file upon completion.

Usage:
    python run_matcha_parallel.py \
        --inference_script inference_matcha_llava.py \
        --input_jsonl /path/to/your/matcha_vqa_tem_only.jsonl \
        --image_dir /path/to/your/images \
        --output_dir /path/to/your/outputs \
        --num_gpus 4
"""

import subprocess
import json
import math
import os
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--inference_script', type=str, required=True,
                    help='Inference script to run (e.g. inference_matcha_llava.py)')
parser.add_argument('--input_jsonl', type=str, required=True,
                    help='Path to input JSONL file')
parser.add_argument('--image_dir',   type=str, required=True,
                    help='Directory containing MatCha images')
parser.add_argument('--output_dir',  type=str, required=True,
                    help='Directory to save output JSONL files')
parser.add_argument('--num_gpus',    type=int, default=4,
                    help='Number of GPUs to use (default: 4)')
args = parser.parse_args()

# Derive model tag from script name
script_name = os.path.basename(args.inference_script).replace('.py', '')
model_tag   = script_name.replace('inference_matcha_', '')

# Load and split records
records = []
with open(args.input_jsonl, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(line)

total      = len(records)
chunk_size = math.ceil(total / args.num_gpus)

ranges = []
for i in range(args.num_gpus):
    start = i * chunk_size
    end   = min(start + chunk_size, total)
    ranges.append((start, end))

print(f"Script : {args.inference_script}")
print(f"Total  : {total} records")
print(f"Split into {args.num_gpus} chunks: {ranges}")

os.makedirs(args.output_dir, exist_ok=True)

# Launch subprocesses
processes   = []
chunk_paths = []

for i, (start, end) in enumerate(ranges):
    chunk_output = os.path.join(
        args.output_dir,
        f"matcha_{model_tag}_chunk{i}_{start}_{end}.jsonl"
    )
    chunk_paths.append(chunk_output)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(i)

    cmd = [
        "python", args.inference_script,
        "--input_jsonl",  args.input_jsonl,
        "--image_dir",    args.image_dir,
        "--output_jsonl", chunk_output,
        "--start",        str(start),
        "--end",          str(end),
    ]

    print(f"Launching GPU {i}: rows {start}~{end} → {os.path.basename(chunk_output)}")
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)

# Wait for all subprocesses
for i, p in enumerate(processes):
    p.wait()
    print(f"GPU {i} finished with return code {p.returncode}")

# Merge results
print("\nMerging results...")

id_order  = [json.loads(line)['id'] for line in records]
id_to_idx = {id_: i for i, id_ in enumerate(id_order)}

all_outputs = []
for chunk_path in chunk_paths:
    if not os.path.exists(chunk_path):
        print(f"WARNING: {chunk_path} not found!")
        continue
    with open(chunk_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_outputs.append(json.loads(line))

all_outputs.sort(key=lambda x: id_to_idx.get(x['id'], 0))

final_output = os.path.join(
    args.output_dir,
    f"matcha_{model_tag}_outputs.jsonl"
)

with open(final_output, 'w', encoding='utf-8') as f:
    for out in all_outputs:
        f.write(json.dumps(out, ensure_ascii=False) + '\n')

print(f"Merged {len(all_outputs)} records → {final_output}")

# Remove chunk files
for chunk_path in chunk_paths:
    if os.path.exists(chunk_path):
        os.remove(chunk_path)
        print(f"Removed: {os.path.basename(chunk_path)}")

# Prediction distribution
print("\n=== Prediction distribution ===")
pred_counter = Counter()
ans_counter  = Counter()

for r in all_outputs:
    for pred in r.get('prediction', []):
        pred_counter[pred.strip()] += 1
    for vqa in r.get('vqa', []):
        ans_counter[vqa.get('answer', '')] += 1

total_pred = sum(pred_counter.values())
total_ans  = sum(ans_counter.values())

print("Predictions:")
for k in sorted(pred_counter.keys()):
    v = pred_counter[k]
    print(f"  {k}: {v:4d} ({v/total_pred*100:.1f}%)")

print("\nGround truth answers:")
for k in sorted(ans_counter.keys()):
    v = ans_counter[k]
    print(f"  {k}: {v:4d} ({v/total_ans*100:.1f}%)")

print(f"\nAll done! Output: {final_output}")
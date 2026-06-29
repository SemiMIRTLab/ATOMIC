"""
Run TEM-MCQ inference in parallel across multiple GPUs.

Each GPU processes a separate chunk of the input CSV.
Results are merged into a single output file upon completion.

Usage:
    python run_TEM_MCQ_parallel.py \
        --inference_script inference_TEM_MCQ_llava.py \
        --input_csv /path/to/your/TEM_MCQ.csv \
        --image_dir /path/to/your/TEM_figures \
        --output_dir /path/to/your/outputs \
        --num_gpus 4
"""

import subprocess
import pandas as pd
import os
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inference_script', type=str, required=True,
                    help='Inference script to run (e.g. inference_TEM_MCQ_llava.py)')
parser.add_argument('--input_csv',  type=str, required=True,
                    help='Path to input CSV (TEM_MCQ.csv)')
parser.add_argument('--image_dir',  type=str, required=True,
                    help='Directory containing TEM subfigure images')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory to save output CSV files')
parser.add_argument('--num_gpus',   type=int, default=4,
                    help='Number of GPUs to use (default: 4)')
args = parser.parse_args()

# Derive model tag from script name
script_name = os.path.basename(args.inference_script).replace('.py', '')
model_tag   = script_name.replace('inference_TEM_MCQ_', '')

# Split input CSV into chunks
df         = pd.read_csv(args.input_csv, dtype=str)
total      = len(df)
chunk_size = math.ceil(total / args.num_gpus)

ranges = []
for i in range(args.num_gpus):
    start = i * chunk_size
    end   = min(start + chunk_size, total)
    ranges.append((start, end))

print(f"Script : {args.inference_script}")
print(f"Total  : {total} rows")
print(f"Split into {args.num_gpus} chunks: {ranges}")

os.makedirs(args.output_dir, exist_ok=True)

# Launch subprocesses
processes   = []
chunk_paths = []

for i, (start, end) in enumerate(ranges):
    chunk_output = os.path.join(
        args.output_dir,
        f"TEM_MCQ_{model_tag}_chunk{i}_{start}_{end}.csv"
    )
    chunk_paths.append(chunk_output)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(i)

    cmd = [
        "python", args.inference_script,
        "--input_csv",  args.input_csv,
        "--image_dir",  args.image_dir,
        "--output_csv", chunk_output,
        "--start",      str(start),
        "--end",        str(end),
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
files = [cp for cp in chunk_paths if os.path.exists(cp)]

if not files:
    print("No output files found.")
else:
    merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    final_output = os.path.join(args.output_dir, f"TEM_MCQ_{model_tag}_outputs.csv")
    merged.to_csv(final_output, index=False)
    print(f"Merged {len(files)} files → {len(merged)} rows")
    print(f"Saved to: {final_output}")

# Remove chunk files
for chunk_path in chunk_paths:
    if os.path.exists(chunk_path):
        os.remove(chunk_path)
        print(f"Removed: {os.path.basename(chunk_path)}")

print("\nAll done!")
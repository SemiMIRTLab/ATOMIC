import torch
import gc
import argparse
from pathlib import Path
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from utils_tem_mcq import prompt, pil_format, answer_parser, load_csv, write_row, init_csv

# Gemma3 baseline
# MODEL_PATH = "google/gemma-3-4b-it"

# ATOMIC-Gemma
MODEL_PATH = "/path/to/your/ATOMIC-gemma"

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv',  type=str, required=True)
parser.add_argument('--image_dir',  type=str, required=True)
parser.add_argument('--output_csv', type=str, required=True)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end',   type=int, required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model: {MODEL_PATH}")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print(f"Model loaded")

df        = load_csv(args.input_csv, args.start, args.end)
image_dir = Path(args.image_dir)
init_csv(args.output_csv)
print(f"Processing rows {args.start}~{args.end-1}, total: {len(df)}")

for idx, row in df.iterrows():
    image_name     = row["CROP_IMAGE"]
    image_type     = row["image_type"]
    task_type      = row["task_type"]
    correct_answer = row["correct_answer"]
    question       = row["question"]

    print(f"[{args.start}~{args.end}] Processing {idx}: {image_name}")
    image_path = image_dir / image_name

    try:
        image       = pil_format(str(image_path))
        prompt_text = prompt(
            question,
            row["option_A"],
            row["option_B"],
            row["option_C"],
            row["option_D"]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt_text}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

        generation = generation[0][input_len:]
        raw = processor.decode(generation, skip_special_tokens=True).strip()
        raw    = raw.split("\n")[0].strip()
        parsed = answer_parser(raw)
        print(f"  raw: {raw!r}  parsed: {parsed!r}  ans: {correct_answer}")
        write_row(args.output_csv, [image_name, image_type, task_type, parsed, correct_answer])

    except Exception as e:
        import traceback; traceback.print_exc()
        write_row(args.output_csv, [image_name, image_type, task_type, f"ERROR: {e}", correct_answer])
        continue

    try:
        del image, inputs, generation
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

print(f"Done! Saved to {args.output_csv}")
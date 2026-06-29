import torch
import gc
import argparse
from pathlib import Path
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from utils_matcha import prompt, crop_subfigure, answer_parser, load_records, save_results

# Gemma3 baseline
# MODEL_PATH = "google/gemma-3-4b-it"

# ATOMIC-Gemma
MODEL_PATH = "/path/to/your/ATOMIC-gemma"

parser = argparse.ArgumentParser()
parser.add_argument('--input_jsonl',  type=str, required=True)
parser.add_argument('--output_jsonl', type=str, required=True)
parser.add_argument('--image_dir',    type=str, required=True)
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

records   = load_records(args.input_jsonl, args.start, args.end)
image_dir = Path(args.image_dir)
print(f"Processing rows {args.start}~{args.end-1}, total: {len(records)}")

results = []

for record in records:
    entry_id = record['id']
    vqa_list = record['vqa']
    img_info = record['images'][0]
    img_path = str(image_dir / img_info['image_path'])
    geometry = img_info.get('geometry', None)

    print(f"Processing: {entry_id}")

    try:
        image = crop_subfigure(img_path, geometry)
    except Exception as e:
        print(f"  [ERROR] image: {e}")
        out = record.copy()
        out['prediction'] = ['ERROR'] * len(vqa_list)
        out['correct']    = [False]   * len(vqa_list)
        results.append(out)
        continue

    predictions     = []
    is_correct_list = []

    for vqa in vqa_list:
        question       = vqa['question']
        correct_answer = vqa['answer']
        options        = vqa.get('options', {})

        prompt_text = prompt(question)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt_text}
                ]
            }
        ]

        try:
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
            parsed = answer_parser(raw, options)

        except Exception as e:
            import traceback; traceback.print_exc()
            raw, parsed = f"ERROR: {e}", "ERROR"

        is_correct = parsed.strip().lower() == correct_answer.strip().lower()
        predictions.append(parsed)
        is_correct_list.append(is_correct)

        print(f"  raw   : {raw!r}")
        print(f"  parsed: {parsed!r}  ans: {correct_answer}  ok: {is_correct}")

        try:
            del inputs, generation
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    out = record.copy()
    out['prediction'] = predictions
    out['correct']    = is_correct_list
    results.append(out)

    del image
    gc.collect()
    torch.cuda.empty_cache()

save_results(results, args.output_jsonl)
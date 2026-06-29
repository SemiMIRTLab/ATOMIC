import torch
import gc
import argparse
from pathlib import Path

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from utils_matcha import prompt, crop_subfigure, answer_parser, load_records, save_results

# LLaVA baseline
# MODEL_PATH = "liuhaotian/llava-v1.5-7b"

# ATOMIC-LLaVA
MODEL_PATH = "/path/to/your/ATOMIC-llava"

parser = argparse.ArgumentParser()
parser.add_argument('--input_jsonl',  type=str, required=True)
parser.add_argument('--output_jsonl', type=str, required=True)
parser.add_argument('--image_dir',    type=str, required=True)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end',   type=int, required=True)
args = parser.parse_args()

device     = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "llava-v1.5-7b"
print(f"Loading model: {MODEL_PATH}")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, model_name
)
model.eval()
print(f"Model loaded on {device}")

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
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

        try:
            images_tensor = process_images(
                [image], image_processor, model.config
            ).to(device, dtype=torch.float16)

            input_ids = tokenizer_image_token(
                qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=[image.size],
                    max_new_tokens=16,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )

            raw    = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
            del images_tensor, input_ids, output_ids
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
import torch
import gc
import argparse
from pathlib import Path

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from utils_tem_vqa import prompt, pil_format, load_csv, write_row, init_csv

# LLaVA baseline
# MODEL_PATH = "liuhaotian/llava-v1.5-7b"

# ATOMIC-LLaVA
MODEL_PATH = "/path/to/your/ATOMIC-llava"

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv',  type=str, required=True)
parser.add_argument('--image_dir',  type=str, required=True)
parser.add_argument('--output_csv', type=str, required=True)
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
        prompt_text = prompt(question)
        qs          = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

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
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

        raw = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"  raw: {raw!r}  ans: {correct_answer}")
        write_row(args.output_csv, [image_name, image_type, task_type, raw, correct_answer])

    except Exception as e:
        import traceback; traceback.print_exc()
        write_row(args.output_csv, [image_name, image_type, task_type, f"ERROR: {e}", correct_answer])
        continue

    del image, images_tensor, input_ids, output_ids
    gc.collect()
    torch.cuda.empty_cache()

print(f"Done! Saved to {args.output_csv}")
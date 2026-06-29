import re
import json
import numpy as np
from PIL import Image
import tifffile as tf


# ===== Prompt =====
def prompt(question_with_options: str) -> str:
    return (
        "You are a visual question answering assistant.\n"
        "Look at the image carefully and answer the following multiple-choice question.\n"
        "You must answer with ONLY ONE LETTER corresponding to the correct option.\n\n"
        f"Question: {question_with_options}\n\n"
        "Answer:"
    )


# ===== Image loading =====
def pil_format(img_path: str) -> Image.Image:
    if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        return Image.open(img_path).convert("RGB")
    elif img_path.lower().endswith(('.tif', '.tiff')):
        try:
            data = tf.imread(img_path)
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            mn, mx = np.min(data), np.max(data)
            normalized = (data - mn) / (mx - mn) * 255 if mx > mn else np.zeros_like(data)
            return Image.fromarray(normalized.astype(np.uint8)).convert("RGB")
        except Exception:
            return Image.open(img_path).convert("RGB")
    else:
        return Image.open(img_path).convert("RGB")


def crop_subfigure(img_path: str, geometry=None) -> Image.Image:
    image = pil_format(img_path)
    if geometry is None:
        return image
    xs = [int(pt["x"]) for pt in geometry]
    ys = [int(pt["y"]) for pt in geometry]
    x1 = max(min(xs), 0); x2 = min(max(xs), image.width)
    y1 = max(min(ys), 0); y2 = min(max(ys), image.height)
    return image.crop((x1, y1, x2, y2))


# ===== Answer parser =====
def answer_parser(raw: str, options: dict) -> str:
    pred = raw.strip()

    # CoT trigger
    for trigger in ["the answer's letter is", "answer's letter is"]:
        if trigger in pred.lower():
            pred = pred.split(trigger)[-1].strip()
            break

    # Common answer patterns
    for trigger in [
        "the correct answer is",
        "the answer is",
        "correct answer:",
        "answer:",
    ]:
        if trigger in pred.lower():
            pred = pred.lower().split(trigger)[-1].strip()
            break

    # Remove noise phrases
    for phrase in ["I understand", "A through B", "A through C",
                   "A through D", "A through E", "A through F", "A through G"]:
        pred = pred.replace(phrase, "")

    # Match valid option letter
    valid = [chr(65 + i) for i in range(len(options))]
    options_str = r"\b([" + "".join(valid) + "".join(v.lower() for v in valid) + r"])\b"
    found = re.findall(options_str, pred)
    if found:
        result = found[0].upper()
        return result[:-1] if result.endswith(".") else result
    return pred.strip()


# ===== Data loading =====
def load_records(input_jsonl: str, start: int, end: int) -> list:
    records = []
    with open(input_jsonl, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(__import__('json').loads(line))
    return records[start:end]


# ===== Output =====
def save_results(results: list, output_jsonl: str):
    import json, os
    os.makedirs(os.path.dirname(output_jsonl) or '.', exist_ok=True)
    with open(output_jsonl, "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    total   = sum(len(r['correct']) for r in results)
    correct = sum(c for r in results for c in r['correct'] if c)
    print(f"\nDone! Saved to {output_jsonl}")
    if total > 0:
        print(f"Quick check — {correct}/{total} = {correct/total*100:.2f}%")
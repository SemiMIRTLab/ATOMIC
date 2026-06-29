import re
import csv
import os
import pandas as pd
import numpy as np
from PIL import Image


# ===== Prompt =====
def prompt(question: str, a: str, b: str, c: str, d: str) -> str:
    return (
        "You are a visual question answering assistant.\n"
        "Look at the image carefully and answer the following multiple-choice question.\n"
        "You must answer with ONLY ONE LETTER: A, B, C, or D.\n\n"
        f"Question: {question}\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n\n"
        "Answer:"
    )


# ===== Image loading =====
def pil_format(img_path: str) -> Image.Image:
    if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        return Image.open(img_path).convert("RGB")
    elif img_path.lower().endswith(('.tif', '.tiff')):
        try:
            import tifffile as tf
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


# ===== Answer parser =====
def answer_parser(raw: str, n_options: int = 4) -> str:
    pred = raw.strip()

    # CoT trigger
    for trigger in ["the answer's letter is", "answer's letter is"]:
        if trigger in pred.lower():
            pred = pred.split(trigger)[-1].strip()
            break

    # Common answer patterns
    for trigger in ["the correct answer is", "the answer is",
                    "correct answer:", "answer:"]:
        if trigger in pred.lower():
            pred = pred.lower().split(trigger)[-1].strip()
            break

    # Remove noise phrases
    for phrase in ["I understand", "A through B", "A through C",
                   "A through D", "A through E", "A through F", "A through G"]:
        pred = pred.replace(phrase, "")

    # Match valid option letter
    valid = [chr(65 + i) for i in range(n_options)]
    options_str = r"\b([" + "".join(valid) + "".join(v.lower() for v in valid) + r"])\b"
    found = re.findall(options_str, pred)
    if found:
        result = found[0].upper()
        return result[:-1] if result.endswith(".") else result
    return pred.strip()


# ===== Data loading =====
def load_csv(csv_path: str, start: int, end: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8")
    return df.iloc[start:end].reset_index(drop=True)


# ===== Output =====
def init_csv(output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["CROP_IMAGE", "IMAGE_TYPE", "TASK_TYPE", "RESPONSE", "CORRECT_ANSWER"])


def write_row(output_path: str, row: list):
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
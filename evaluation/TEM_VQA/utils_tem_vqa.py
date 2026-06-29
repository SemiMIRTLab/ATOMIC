import re
import csv
import os
import pandas as pd
import numpy as np
from PIL import Image


# ===== Prompt =====
def prompt(question: str) -> str:
    return (
        "You are a visual question answering assistant.\n"
        "Look at the image carefully and answer the following question.\n"
        "Provide a concise and accurate answer based on what you observe in the image.\n\n"
        f"Question: {question}\n\nAnswer:"
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
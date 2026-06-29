"""
Gemma-3-4b-it Vision SFT Training Script

Usage:
  torchrun --nproc_per_node=4 train_gemma3_vision.py
  torchrun --nproc_per_node=4 train_gemma3_vision.py --test
"""

import json
import os
import sys
from typing import Any

import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

# ── Test mode ────────────────────────────────────────────────
TEST_MODE = "--test" in sys.argv

# ── Path settings ────────────────────────────────────────────
MODEL_ID   = "google/gemma-3-4b-it"
DATA_PATH  = "/path/to/your/Conversion.json"   # Stage 2 data converted to Gemma format
IMAGE_DIR  = "/path/to/your/TEM_figures"
OUTPUT_DIR = (
    "/path/to/your/checkpoints_test"
    if TEST_MODE else
    "/path/to/your/ATOMIC-gemma"
)

# Gemma3 assistant turn token IDs
# <start_of_turn>model → [105, 4368]
# <end_of_turn>        → [106]
ASSISTANT_START_TOKENS = [105, 4368]
END_OF_TURN_TOKEN      = 106


def mask_user_tokens(labels: torch.Tensor) -> torch.Tensor:
    """
    Retain loss only for assistant response tokens.
    Sets user prompt token labels to -100.

    Logic:
    - Default: mask all tokens (-100)
    - On <start_of_turn>model: begin recording (is_assistant = True)
    - On <end_of_turn>: stop recording, mask this token itself
    """
    result = labels.clone()
    result[:] = -100

    for i in range(labels.shape[0]):
        seq = labels[i].tolist()
        is_assistant = False
        start_match = 0

        j = 0
        while j < len(seq):
            token = seq[j]

            if not is_assistant:
                if token == ASSISTANT_START_TOKENS[start_match]:
                    start_match += 1
                    if start_match == len(ASSISTANT_START_TOKENS):
                        is_assistant = True
                        start_match = 0
                else:
                    start_match = 0
            else:
                if token == END_OF_TURN_TOKEN:
                    is_assistant = False
                elif token == -100:
                    pass
                else:
                    result[i][j] = labels[i][j]
            j += 1

    return result


# ── Data loading ─────────────────────────────────────────────
def load_dataset_from_json(data_path: str, image_dir: str, limit: int = None) -> Dataset:
    print(f"Loading data: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    valid_data = []
    skipped = 0
    for item in raw_data:
        img_path = os.path.join(image_dir, os.path.basename(item["image"]))
        if os.path.exists(img_path):
            item["image_path"] = img_path
            valid_data.append(item)
        else:
            skipped += 1

    if limit:
        valid_data = valid_data[:limit]

    print(f"Valid samples: {len(valid_data)}, skipped: {skipped} (image not found)")
    return Dataset.from_list(valid_data)


# ── Collator ─────────────────────────────────────────────────
class Gemma3VisionCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, samples: list[dict[str, Any]]):
        batch_texts = []
        batch_images = []

        for sample in samples:
            img = Image.open(sample["image_path"]).convert("RGB")

            messages = []
            for msg in sample["messages"]:
                new_msg = {"role": msg["role"], "content": []}
                for content in msg["content"]:
                    if content["type"] == "image":
                        new_msg["content"].append({
                            "type": "image",
                            "image": img
                        })
                    else:
                        new_msg["content"].append(content)
                messages.append(new_msg)

            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            batch_texts.append(text)
            batch_images.append(img)

        batch = self.processor(
            text=batch_texts,
            images=[[img] for img in batch_images],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Mask padding and image tokens
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        labels[labels == image_token_id] = -100

        # Retain loss only for assistant response tokens
        labels = mask_user_tokens(labels)

        batch["labels"] = labels
        return batch


# ── Main training loop ───────────────────────────────────────
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"{'[TEST MODE] ' if TEST_MODE else ''}OUTPUT_DIR: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )

    # Freeze vision encoder (vision_tower)
    # Train connector (multi_modal_projector) + LLM (language_model)
    print("Setting trainable parameters...")
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable / 1e9:.2f}B / {total / 1e9:.2f}B ({trainable/total*100:.1f}%)")

    dataset = load_dataset_from_json(
        DATA_PATH, IMAGE_DIR,
        limit=10 if TEST_MODE else None
    )

    collator = Gemma3VisionCollator(processor, max_length=2048)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1 if TEST_MODE else 3,
        per_device_train_batch_size=2 if TEST_MODE else 16,
        gradient_accumulation_steps=1 if TEST_MODE else 1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1 if TEST_MODE else 10,
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
"""
Convert LLaVA JSON format to Gemma3 Vision chat template format.

LLaVA format:
{
    "id": "...",
    "image": "TEM_figures/xxx.png",
    "conversations": [
        {"from": "human", "value": "<image>\nQuestion?"},
        {"from": "gpt", "value": "Answer."},
        ...
    ]
}

Gemma3 format:
{
    "id": "...",
    "image": "TEM_figures/xxx.png",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Question?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Answer."}]
        },
        ...
    ]
}

Usage:
    python convert_to_gemma_format.py --input /path/to/input.json --output /path/to/output.json
"""

import json
import argparse
from pathlib import Path


def convert_conversation(conversations: list) -> list:
    """Convert LLaVA conversations to Gemma3 messages format."""
    messages = []

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]

        if role == "user":
            content = []
            if "<image>" in text:
                text = text.replace("<image>", "").strip()
                content.append({"type": "image"})
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": text}]
            })

    return messages


def convert_dataset(input_path: str, output_path: str):
    """Convert entire dataset from LLaVA to Gemma3 format."""

    print(f"Loading data: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}, starting conversion...")

    converted = []
    skipped = 0

    for item in data:
        if "conversations" not in item:
            skipped += 1
            continue

        converted_item = {
            "id": item.get("id", ""),
            "image": item.get("image", ""),
            "messages": convert_conversation(item["conversations"])
        }
        converted.append(converted_item)

    print(f"Conversion complete: {len(converted)} succeeded, {skipped} skipped")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")

    print("\n=== First sample (converted) ===")
    print(json.dumps(converted[0], ensure_ascii=False, indent=2))

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaVA format to Gemma3 Vision format"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    convert_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
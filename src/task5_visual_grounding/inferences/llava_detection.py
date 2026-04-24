"""LLaVA Next inference."""

import argparse
import json
import os
from typing import Any

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


MODEL_NAME = "llava-hf/llava-v1.6-vicuna-7b-hf"


def run_inference(processed_file: str, images_dir: str, results_file: str, cache_dir: str) -> None:
    """Run inference."""
    with open(processed_file, "r") as f:
        processed_data = json.load(f)

    results = []

    model = (
        LlavaNextForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, cache_dir=cache_dir)
        .to("cuda" if torch.cuda.is_available() else "cpu")
        .eval()
    )

    processor = LlavaNextProcessor.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

    def build_prompt(question: str) -> str:
        return (
            "<image>\n"
            f"{question}\n\n"
            "Please return only the bounding box coordinates that answer this question "
            "in the exact format: x1,y1,x2,y2 (no extra text)."
        )

    def run_model(prompt: str, image: Any = None) -> str:
        inputs = processor(text=[prompt], images=image if image else None, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)

        return processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    for i, sample in enumerate(processed_data, 1):
        sample_id = sample["id"]
        question = sample["question"].strip()
        gt_bbox = sample["bbox"]

        prompt = build_prompt(question)

        img_path = os.path.join(images_dir, sample_id)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{i}] Failed to load image {sample_id}: {e}")
            continue

        try:
            output_text = run_model(prompt, image=image)
        except Exception as e:
            output_text = f"[GENERATION_FAILED: {str(e)}]"

        coords = []
        for part in output_text.replace(",", " ").split():
            try:
                coords.append(float(part))
            except Exception:
                continue
        pred_bbox = coords[:4] if len(coords) >= 4 else None

        results.append(
            {
                "id": sample_id,
                "question": question,
                "ground_truth_bbox": gt_bbox,
                "model_output": output_text,
                "predicted_bbox": pred_bbox,
            }
        )
        print(f"[{i}/{len(processed_data)}] {sample_id} → {pred_bbox if pred_bbox else 'None'}")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Done! {len(results)} new samples processed.")


if __name__ == "__main__":
    """Run inference with LLaVA Next model."""
    parser = argparse.ArgumentParser(description="Run inference with LLaVA Next model.")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the processed dataset JSON file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the directory containing the images.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the results JSON file.")
    parser.add_argument("--cache_dir", type=str, default="", help="Cache directory for model and processor.")

    args = parser.parse_args()
    MODEL_NAME = args.model_name  # Global override

    run_inference(
        processed_file=args.processed_file,
        images_dir=args.images_dir,
        results_file=args.results_file,
        cache_dir=args.cache_dir,
    )

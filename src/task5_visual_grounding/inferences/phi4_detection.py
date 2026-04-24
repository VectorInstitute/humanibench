"""HumanI Bench — task 5: Phi-4 visual grounding detection."""

import argparse
import json
import os
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"


def run_inference(processed_file: str, images_dir: str, results_file: str, cache_dir: str) -> None:
    """Run inference."""
    with open(processed_file, "r") as f:
        processed_data = json.load(f)

    results = []

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        _attn_implementation="eager",
        cache_dir=cache_dir,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=cache_dir)

    # Ensure <image> is a special token
    if "<image>" not in processor.tokenizer.additional_special_tokens:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        model.resize_token_embeddings(len(processor.tokenizer))

    model.eval()

    def build_prompt(question: str) -> str:
        return (
            "<|user|><|image_1|>\n"
            f"{question}\n"
            "Respond with the bounding box in the format:\n"
            "x1,y1,x2,y2 only.\n<|end|><|assistant|>"
        )

    def run_model(prompt: str, image: Any = None) -> str:
        inputs = processor(text=[prompt], images=image, return_tensors="pt").to(model.device)

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
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with Phi-4 multimodal model.")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the processed dataset JSON file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the directory containing the images.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the results JSON file.")
    parser.add_argument("--cache_dir", type=str, default="", help="Cache directory for model and processor.")

    args = parser.parse_args()

    run_inference(
        processed_file=args.processed_file,
        images_dir=args.images_dir,
        results_file=args.results_file,
        cache_dir=args.cache_dir,
    )

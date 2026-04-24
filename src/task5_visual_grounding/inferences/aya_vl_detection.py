"""HumanI Bench — task 5: Aya vision bounding-box detection."""

import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_NAME = "CohereForAI/aya-vision-8b"


def run_inference(processed_file: str, images_dir: str, results_file: str, cache_dir: str) -> None:
    """Run inference."""
    with open(processed_file, "r") as f:
        processed_data = json.load(f)

    results = []

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True, cache_dir=cache_dir
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=cache_dir)

    n_done = 0
    for sample in processed_data:
        sample_id = sample["id"]
        question = sample["question"].strip()
        gt_bbox = sample["bbox"]

        prompt = (
            "<image>\n"
            f"{question}\n\n"
            "Please return only the bounding box coordinates that answer this question "
            "in the exact format: x1,y1,x2,y2 (no extra text)."
        )

        img_path = os.path.join(images_dir, sample_id)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {sample_id}: {e}")
            continue

        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
            output_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
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
        n_done += 1
        print(f"[{n_done}/{len(processed_data)}] {sample_id} → {pred_bbox if pred_bbox else 'None'}")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Done! {n_done} new samples processed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with LLaVA-style vision-language model.")
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

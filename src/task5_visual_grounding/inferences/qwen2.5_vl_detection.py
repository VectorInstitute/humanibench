"""HumanI Bench — task 5: Qwen2.5-VL visual grounding detection."""

import argparse
import json
import os

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def run_inference(processed_file: str, images_dir: str, results_file: str, cache_dir: str | None) -> None:
    # Load dataset
    """Run inference."""
    with open(processed_file, "r") as f:
        processed_data = json.load(f)

    results = []

    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

    # Inference loop
    n_done = 0
    for sample in processed_data:
        sample_id = sample["id"]
        question = sample["question"].strip()
        gt_bbox = sample["bbox"]

        prompt = (
            question + "\n\nPlease **only** return the bounding box coordinates that answer this question, "
            "in the exact format: x1,y1,x2,y2 (no extra text)."
        )

        img_path = os.path.join(images_dir, sample_id)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {sample_id}: {e}")
            continue

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        try:
            generated = model.generate(**inputs, max_new_tokens=128)
            gen_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
            output_text = processor.batch_decode(
                gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
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

        # Save incrementally
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Done! {n_done} new samples processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL inference for bounding box detection.")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the processed dataset JSON file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the directory containing the images.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the results JSON file.")
    parser.add_argument("--cache_dir", type=str, help="Cache directory for model and processor.")

    args = parser.parse_args()

    run_inference(
        processed_file=args.processed_file,
        images_dir=args.images_dir,
        results_file=args.results_file,
        cache_dir=args.cache_dir,
    )

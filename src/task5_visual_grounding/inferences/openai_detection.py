"""HumanI Bench — task 5 (visual grounding): openai detection."""

import base64
import json
import os

import openai
from PIL import Image


def run_inference(processed_file: str, images_dir: str, results_file: str, cache_dir: str | None = None) -> None:
    """Run inference."""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def build_prompt(question: str) -> str:
        return question

    def run_openai_vision(prompt: str, image_path: str) -> str:
        base64_image = encode_image(image_path)
        messages = [
            {"role": "system", "content": "You are a helpful image detection assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]
        try:
            response = openai.ChatCompletion.create(model="gpt-4o", messages=messages, max_tokens=100, temperature=0.0)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[GENERATION_FAILED: {str(e)}]"

    with open(processed_file, "r") as f:
        processed_data = json.load(f)

    results = []
    for i, sample in enumerate(processed_data, 1):
        sample_id = sample["id"]
        question = sample["question"].strip()
        gt_bbox = sample["bbox"]

        prompt = build_prompt(question)
        image_path = os.path.join(images_dir, sample_id)

        try:
            _ = Image.open(image_path).convert("RGB")  # Just to validate image exists
        except Exception as e:
            print(f"[{i}] Failed to load image {sample_id}: {e}")
            continue

        output_text = run_openai_vision(prompt, image_path)

        coords = []
        for part in output_text.replace(",", " ").split():
            try:
                coords.append(float(part))
            except ValueError:
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

    parser = argparse.ArgumentParser(description="Run OpenAI GPT-4o vision inference.")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the processed dataset JSON file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the directory containing the images.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the results JSON file.")
    parser.add_argument("--cache_dir", type=str, default="", help="(Unused, for compatibility)")

    args = parser.parse_args()

    run_inference(
        processed_file=args.processed_file,
        images_dir=args.images_dir,
        results_file=args.results_file,
        cache_dir=args.cache_dir,
    )

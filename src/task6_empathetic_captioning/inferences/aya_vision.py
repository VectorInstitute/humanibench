#!/usr/bin/env python
"""HumanI Bench — task 6 (empathetic captioning): aya vision."""

import csv
import glob
import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any

import torch
from huggingface_hub import login
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def main(args: Namespace) -> None:
    # Log in to Hugging Face
    """Run the main workflow."""
    login(token=args.hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    # Load CSV data and create automatic selected samples
    csv_captions = {}
    auto_selected_samples = []
    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = os.path.splitext(row["Image"])[0]  # Remove extension
            csv_captions[image_id] = {"csv_simple": row["simple_Caption"], "csv_emphatic": row["emphatic_Caption"]}
            auto_selected_samples.append({"id": image_id})

    # Get all image paths
    image_files = glob.glob(os.path.join(args.image_folder, "*.*"))
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_paths = [f for f in image_files if os.path.splitext(f)[1].lower() in image_extensions]

    # Create ID to path mapping
    id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}

    # Load model
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.float16
    ).to(device)

    # Define caption prompts
    prompts = {
        "model_simple": "Describe this image concisely and objectively.",
        "model_empathetic": "Describe this image in a compassionate, human-centered way...",
    }

    results = []

    # Process each sample from CSV
    for sample in auto_selected_samples:
        sample_id = str(sample["id"])
        image_path = id_to_path.get(sample_id)

        if not image_path:
            print(f"Skipping sample {sample_id}: Image not found in folder")
            continue

        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Skipping {sample_id}: Error loading image - {str(e)}")
            continue

        # Initialize result entry
        result_entry: dict[str, Any] = {
            "id": sample_id,
            "image_path": image_path,
            "csv_captions": csv_captions.get(sample_id, {}),
            "model_captions": {},
        }

        # Generate model captions
        for caption_type, prompt in prompts.items():
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
            ]

            inputs = processor.apply_chat_template(
                messages,
                padding=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )

            caption = processor.tokenizer.decode(
                gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

            result_entry["model_captions"][caption_type] = caption

        results.append(result_entry)

        # Save incremental results
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Processed {len(results)} images")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference with Aya Vision and save captions.")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace authentication token")
    parser.add_argument("--model_path", type=str, default="CohereForAI/aya-vision-8b", help="Model ID for Aya-Vision")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to your combined.csv")
    parser.add_argument("--results_file", type=str, default="./results/caption_results.json", help="Output JSON file")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    args = parser.parse_args()

    main(args)

# This script is designed to run inference using the Aya Vision model from Hugging Face.
# It processes images, generates captions based on prompts, and saves the results in a JSON file.

# To run the script, use the following command:
# python aya_vision.py \
#     --hf_token <your_huggingface_token> \
#     --model_path CohereForAI/aya-vision-8b \
#     --csv_file <path_to_your_combined_csv> \
#     --results_file <path_to_your_results_json> \
#     --image_folder <path_to_your_image_folder>

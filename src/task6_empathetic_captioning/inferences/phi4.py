#!/usr/bin/env python
"""HumanI Bench — task 6 (empathetic captioning): phi4."""

import csv
import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any

import torch
from huggingface_hub import login
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


# Set environment variables for caching
os.environ["HF_HOME"] = ""  # Path where you want to store the huggingface cache
os.environ["TRANSFORMERS_CACHE"] = ""  # Path where you want to store the transformers cache


def main(args: Namespace) -> None:
    # Log in to Hugging Face
    """Run the main workflow."""
    login(token=args.hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Phi-4 multimodal components
    processor = AutoProcessor.from_pretrained(args.phi4_model, cache_dir=os.environ["HF_HOME"], trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.phi4_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        cache_dir=os.environ["HF_HOME"],
        _attn_implementation="eager",
    ).eval()

    # Load CSV data
    csv_data = {}
    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = os.path.splitext(row["Image"])[0]
            csv_data[image_id] = {"simple": row["simple_Caption"], "emphatic": row["emphatic_Caption"]}

    results = []
    for image_id, captions in csv_data.items():
        print(f"Processing {image_id}...")
        image_path = os.path.join(args.image_folder, f"{image_id}.jpg")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_id}: {str(e)}")
            continue

        # Define prompts using correct template
        prompts = {
            "simple": "<|user|><|image_1|>Describe this image concisely.<|end|><|assistant|>",
            "emphatic": "<|user|><|image_1|>Create an empathetic caption for this image considering the emotional context.<|end|><|assistant|>",
        }

        result_entry: dict[str, Any] = {
            "id": image_id,
            "csv_captions": captions,
            "phi4_captions": {},
        }

        for caption_type, prompt in prompts.items():
            # Format input according to Phi-4 requirements
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            # Generation parameters
            generation_config = GenerationConfig(
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
            )

            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)

            # Decode response properly
            generated_ids = outputs[:, inputs.input_ids.shape[1] :]
            response = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            result_entry["phi4_captions"][caption_type] = response

        results.append(result_entry)

        # Save after each image to prevent data loss
        results_file_path = os.path.join(args.results_folder, args.results_file)
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Completed processing {len(results)} images")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--phi4_model", type=str, default="microsoft/Phi-4-multimodal-instruct")
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./results")
    parser.add_argument("--results_file", type=str, default="phi4_results.json")
    parser.add_argument("--image_folder", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.results_folder, exist_ok=True)
    main(args)

# This script is designed to run inference using the Phi-4 model for image captioning.
# It processes images and generates captions based on the provided CSV file.

# To run the script, use the following command:
# python phi4.py \
#     --hf_token <your_huggingface_token> \
#     --phi4_model microsoft/Phi-4-multimodal-instruct \
#     --csv_file <path_to_your_combined_csv> \
#     --results_folder <path_to_your_results_folder> \
#     --results_file <path_to_your_results_json> \
#     --image_folder <path_to_your_image_folder>

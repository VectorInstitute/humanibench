#!/usr/bin/env python
"""HumanI Bench — task 6 (empathetic captioning): deepseek."""

import csv
import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import Any

import torch
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from huggingface_hub import login
from PIL import Image


# Set environment variables for caching
os.environ["HF_HOME"] = ""  # Path where you want to store the huggingface cache
os.environ["TRANSFORMERS_CACHE"] = ""  # Path where you want to store the transformers cache


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize image to fit within the specified max size."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        print(f"Error resizing image {img_path}: {e}")
        return None


def generate_caption(model: Any, tokenizer: Any, processor: Any, image: Any, prompt: str) -> str:
    """Generate a caption for the given image using the model."""
    try:
        # Create conversation format expected by DeepSeek
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}", "images": [image]},
            {"role": "<|Assistant|>", "content": ""},
        ]

        inputs = processor(conversations=conversation, images=[image], force_batchify=True).to(model.device)
        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
            )
        return tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True).strip()

    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Error"


def main(args: Namespace) -> None:
    # Log in to HuggingFace
    """Run the main workflow."""
    login(token=args.hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if results folder exists, if not create it
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    # Load model
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path, cache_dir=os.environ["HF_HOME"]).to(device)
    tokenizer = processor.tokenizer
    model = (
        DeepseekVLV2ForCausalLM.from_pretrained(args.model_path, cache_dir=os.environ["HF_HOME"])
        .to(torch.bfloat16)
        .to(device)
        .eval()
    )

    # Load CSV data
    csv_captions = {}
    auto_selected_samples = []
    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = os.path.splitext(row["Image"])[0]
            csv_captions[image_id] = {"csv_simple": row["simple_Caption"], "csv_emphatic": row["emphatic_Caption"]}
            auto_selected_samples.append({"id": image_id})

    # Map image ID to path
    image_files = glob(os.path.join(args.image_folder, "*.*"))
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    id_to_path = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in image_files
        if os.path.splitext(p)[1].lower() in image_extensions
    }

    prompts = {
        "model_simple": "Describe this image concisely and objectively.",
        "model_empathetic": "Describe this image in a compassionate, human-centered way.",
    }

    results = []
    for sample in auto_selected_samples:
        sample_id = str(sample["id"])
        image_path = id_to_path.get(sample_id)

        if not image_path:
            print(f"Skipping {sample_id}: Image not found.")
            continue

        image = resize_image(image_path)
        if image is None:
            continue

        result_entry: dict[str, Any] = {
            "id": sample_id,
            "image_path": image_path,
            "csv_captions": csv_captions.get(sample_id, {}),
            "model_captions": {},
        }

        for cap_type, prompt in prompts.items():
            caption = generate_caption(model, tokenizer, processor, image, prompt)
            result_entry["model_captions"][cap_type] = caption

        results.append(result_entry)

        # Save progress incrementally
        results_file_path = os.path.join(args.results_folder, args.results_file)
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Processed {len(results)} images.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run DeepSeek-VL2 inference and save captions.")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace authentication token")
    parser.add_argument(
        "--model_path", type=str, default="deepseek-ai/deepseek-vl2-tiny", help="Model path for DeepSeek"
    )
    parser.add_argument("--csv_file", type=str, required=True, help="Path to combined.csv")
    parser.add_argument("--results_file", type=str, default="deepseek_results.json", help="Output JSON filename")
    parser.add_argument("--results_folder", type=str, default="./results", help="Folder to save results")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    args = parser.parse_args()

    main(args)

# This script is designed to run inference using the DeepSeek-VL2 model from Hugging Face.
# It processes images, generates captions based on prompts, and saves the results in a JSON file.

# To run the script, use the following command:
# python deepseek.py \
#     --hf_token YOUR_HUGGINGFACE_TOKEN \
#     --model_path deepseek-ai/deepseek-vl2-tiny \
#     --csv_file path/to/combined.csv \
#     --results_file deepseek_results.json \
#     --results_folder ./results \
#     --image_folder path/to/images

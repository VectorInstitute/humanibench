"""HumanI Bench — task 1 (scene understanding): generate attributes."""

import base64
import io
import json
from argparse import ArgumentParser
from typing import Any

import openai
from datasets import Features, Value, load_dataset
from datasets import Image as HFImage
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


# Define dataset features
features = Features(
    {
        "unique_id": Value("string"),
        "image": HFImage(),  # Assume images are stored in the dataset
    }
)


# Encode a resized PIL image to Base64
def encode_image_to_base64(image: Any) -> str:
    # Resize image to 256x256
    """Encode image to base64."""
    image = image.resize((256, 256), Image.Resampling.LANCZOS)  # Resize while maintaining quality
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")  # Save the PIL image as JPEG
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"


# Load dataset
def load_hf_dataset(dataset_name: str, split: str) -> Any:
    """Load hf dataset."""
    dataset = load_dataset(dataset_name, split=split)
    print(f"Dataset size: {len(dataset)} entries")
    return dataset


# Collate function for DataLoader
def collate_fn(batch: list[Any]) -> dict[str, list[Any]]:
    """Collate fn."""
    return {"images": [item["image"] for item in batch], "ids": [item["unique_id"] for item in batch]}


# Load existing attributes from JSON
def load_existing_attributes(filepath: str) -> dict[str, Any]:
    """Load existing attributes."""
    try:
        with open(filepath, "r") as f:
            return {item["id"]: item["attributes"] for item in json.load(f)}
    except FileNotFoundError:
        return {}


# Generate attributes using GPT-4
def generate_attributes_openai(
    dataloader: Any, openai_api_key: str, save_path: str, existing_attributes: dict[str, Any]
) -> list[Any]:
    """Generate attributes openai."""
    openai.api_key = openai_api_key
    results: list[Any] = list(existing_attributes.items())  # Start with existing attributes
    processed_ids = set(existing_attributes.keys())

    with tqdm(total=len(dataloader), unit="batch") as pbar:
        for batch in dataloader:
            images, ids = batch["images"], batch["ids"]
            batch_results = []

            for img, unique_id in zip(images, ids):
                if unique_id in processed_ids:
                    continue  # Skip IDs that already have attributes

                try:
                    # Convert PIL image to Base64
                    img_base64 = encode_image_to_base64(img)

                    # Attributes generation prompt
                    prompt_attributes = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Look at the image and identify whether it shows any of the following concepts: Gender, Age, Ethnicity, Sport, Occupation. List only the concepts you can see, separated by commas. If none of these concepts are visible, respond with 'None' and nothing else.",
                            },
                            {"type": "image_url", "image_url": {"url": img_base64}},
                        ],
                    }

                    response_attributes = openai.ChatCompletion.create(
                        model="gpt-4o", messages=[prompt_attributes], max_tokens=50, temperature=0.7
                    )
                    attributes = response_attributes["choices"][0]["message"]["content"].strip()

                    batch_results.append(
                        {
                            "id": unique_id,
                            "attributes": attributes,
                        }
                    )

                except Exception as e:
                    print(f"Error generating attributes for ID {unique_id}: {e}")

            results.extend(batch_results)

            # Save results incrementally after every batch
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)

            pbar.update(1)

    return results


# Argument parsing
def parse_args() -> Any:
    """Parse args."""
    parser = ArgumentParser(description="Generate attributes for images using OpenAI GPT-4.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset")
    parser.add_argument("--split", type=str, required=True, help="The split of the dataset (e.g., 'train', 'test')")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size for processing images")
    parser.add_argument("--openai_api_key", type=str, required=True, help="Your OpenAI API key")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated attributes")
    parser.add_argument("--attributes_file", type=str, required=True, help="Path to existing attributes JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load existing attributes
    existing_attributes = load_existing_attributes(args.attributes_file)

    # Load dataset
    dataset = load_hf_dataset(args.dataset_name, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Generate attributes
    generate_attributes_openai(dataloader, args.openai_api_key, args.save_path, existing_attributes)


# To run this script, use the following command:
# python generate_attributes.py \
#     --dataset_name <dataset_name> \
#     --split <split> \
#     --batch_size <batch_size> \
#     --openai_api_key <your_openai_api_key> \
#     --save_path <path_to_save_attributes> \
#     --attributes_file <path_to_existing_attributes_json>

"""HumanI Bench — task 1 (scene understanding): generate captions descriptions."""

import json
from argparse import ArgumentParser
from typing import Any

import torch
from datasets import Features, Value, load_dataset
from datasets import Image as HFImage
from huggingface_hub import login
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


# Define dataset features
features = Features(
    {
        "unique_id": Value("string"),
        "image": HFImage(),
    }
)


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


# Caption generation function
def generate_captions(model: Any, dataloader: Any, processor: Any, device: Any, save_path: str) -> list[Any]:
    """Generate captions."""
    model.eval()
    results: list[dict[str, Any]] = []

    # Load existing results if the file exists
    try:
        with open(save_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"No existing file found at {save_path}. Starting fresh.")

    # Collect already processed IDs
    processed_ids = {item["id"] for item in results}

    with torch.no_grad(), tqdm(total=len(dataloader), unit="batch") as pbar:
        for batch in dataloader:
            images, ids = batch["images"], batch["ids"]
            batch_results = []
            for img, id in zip(images, ids):
                # Skip IDs that are already processed
                if id in processed_ids:
                    print(f"Skipping ID {id} (already processed).")
                    continue

                try:
                    # Generate caption
                    prompt = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {
                                    "type": "text",
                                    "text": "Provide a descriptive one-sentence caption for the given image: ",
                                },
                            ],
                        }
                    ]

                    input_text = processor.apply_chat_template(prompt, add_generation_prompt=True)
                    inputs = processor(img, input_text, add_special_tokens=False, return_tensors="pt").to(device)

                    output = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.9)
                    caption = processor.decode(output[0], skip_special_tokens=True)

                    # Generate image description
                    prompt = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "Provide a one-paragraph description for the given image: "},
                            ],
                        }
                    ]

                    input_text = processor.apply_chat_template(prompt, add_generation_prompt=True)
                    inputs = processor(img, input_text, add_special_tokens=False, return_tensors="pt").to(device)

                    output = model.generate(**inputs, max_new_tokens=150, temperature=0.7, top_p=0.9)
                    img_description = processor.decode(output[0], skip_special_tokens=True)

                    batch_results.append({"id": id, "caption": caption, "img_description": img_description})

                    # Mark the ID as processed
                    processed_ids.add(id)

                except Exception as e:
                    print(f"Error generating caption for ID {id}: {e}")

            results.extend(batch_results)

            # Save results incrementally after every batch
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)

            pbar.update(1)
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated captions")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace authentication token")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")

    args = parser.parse_args()

    # Login to HuggingFace
    login(token=args.hf_token)

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_hf_dataset(args.dataset_name, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Load model and processor
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    # Generate captions
    generate_captions(model, dataloader, processor, device, args.save_path)

# # To run this script, use the following command:
# # python generate_captions_descriptions.py \
# #     --dataset_name <dataset_name> \
# #     --split <split> \
# #     --batch_size <batch_size> \
# #     --save_path <path_to_save_captions> \
# #     --hf_token <your_huggingface_token> \
# #     --model_path <path_to_model>

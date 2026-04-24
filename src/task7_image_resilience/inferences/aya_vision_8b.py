"""HumanI Bench — task 7: Aya vision 8B eval inference."""

import base64
import json
import logging
import os
import time
from argparse import ArgumentParser
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# from accelerate import Accelerator

# Model directory
MODEL_DIR = "/model-weights/aya-vision-8b"  # Local model path
HF_MODEL_ID = "CohereForAI/aya-vision-8b"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
OFFLOAD_FOLDER = ""


# Load the model and processor
def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load model."""
    print(f"Loading Aya Vision Vision model from {'local directory' if model_source == 'local' else 'Hugging Face'}...")

    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    # accelerator = Accelerator()

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        offload_folder=OFFLOAD_FOLDER,
        trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )

    # The accelerator will handle device placement for you
    # model = accelerator.prepare(model)
    return model, processor


# Resize image
def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize image to fit within max size, keeping aspect ratio."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image  # Returns the resized PIL image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None  # Return None if resizing fails


# Process a single image-question pair and generate an answer
def process_sample(model: Any, processor: Any, img_path: str, question: str, device: Any) -> str:
    """Process sample."""
    try:
        # Resize image but don't save
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        user_prompt = f"Given question, answer in the following format:\
                Question:{question}\
                Answer:<answer> Reasoning:<reasoning> in the context of the image."

        # Convert the image to a byte stream
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")  # Save image as JPEG to the byte stream
        img_bytes = img_byte_arr.getvalue()  # Get the byte data

        # Encode the byte data to base64
        base64_image_url = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_image_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        predicted_answer = processor.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return predicted_answer if predicted_answer else "No answer generated"

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


# Main function to process dataset
def evaluate(
    model: Any,
    processor: Any,
    dataset: list[Any],
    image_folder: str,
    save_path: str,
    attack: str,
) -> None:
    """Evaluate."""
    results = []
    logger.info("Starting evaluation...")
    intermediate_results_path = save_path.replace(".json", "_intermediate.json")
    prev_path = ""
    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            # print(data)
            img_path = os.path.join(image_folder, f"{data['ID']}_{attack}.jpg")

            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue

            answer = process_sample(model, processor, img_path, data["Question"], device)
            results.append(
                {
                    "ID": data["ID"],
                    "Question": data["Question"],
                    "Predicted_Answer": answer,
                    "Ground_Truth": data["Answer"],
                    "Attribute": data["Attribute"],
                }
            )

            # Save intermediate results every 50
            if i % 10 == 0:
                intermediate_results_path = save_path.replace(
                    ".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(intermediate_results_path, "w") as f:
                    json.dump(results, f, indent=4, default=str)
                if prev_path != "":
                    os.remove(prev_path)
                logger.info(f"Intermediate results saved to {intermediate_results_path} and deleted {prev_path}.")
                prev_path = intermediate_results_path

            pbar.update(1)

    # Save results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    logger.info(f"Results saved to {save_path}.")


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--image_folder", type=str, help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--save_path", type=str, default="./results/results_Aya_Vision_8B.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="hf", help="Model source: 'local' or 'hf'")
    parser.add_argument("--attack", type=str, default="compression", help="Attack type")

    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, processor = load_model(args.model_source)
    # model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Select 20 samples for testing
    # dataset = {k: dataset[k] for k in list(dataset.keys())[:20]}

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, args.attack)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

# To run the script:
# python aya_vision_8b.py --dataset <path_to_dataset> \
# --image_folder <path_to_image_folder> \
# --save_path <path_to_save_results> \
# --attack <attack_type>

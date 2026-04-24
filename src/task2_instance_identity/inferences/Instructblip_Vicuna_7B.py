"""HumanI Bench — task 2 (instance identity): Instructblip Vicuna 7B."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor


# Model paths
MODEL_DIR = "/model-weights/instructblip-vicuna-7b/"
HF_MODEL_ID = "Salesforce/instructblip-vicuna-7b"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Hugging Face cache directories
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
offload_folder = ""


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load and return the InstructBLIP model and processor."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading InstructBLIP model from {model_path}...")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"]
    ).eval()
    processor = InstructBlipProcessor.from_pretrained(model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize the image while preserving its aspect ratio."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, processor: Any, img_path: str, question: str) -> str:
    """Generate an answer for a single image-question pair."""
    try:
        device = next(model.parameters()).device
        logger.info(f"Processing {img_path} with question: {question}...")
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        user_prompt = (
            f"Given question, answer in the following format:"
            f"Question:{question}"
            f"Answer:<answer> Reasoning:<reasoning> in the context of the image."
        )

        try:
            inputs = processor(images=image, text=user_prompt, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
        except Exception as e:
            logger.error(f"Error in processor: {e}")
            return f"Error in processor: {e}"

        gen_kwargs = {"max_new_tokens": 150, "do_sample": False, "use_cache": True, "min_length": len(user_prompt)}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
        answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return answer if answer else "No answer"
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str) -> None:
    """Process the dataset and save evaluation results to a JSON file."""
    results = []
    logger.info("Starting evaluation...")
    prev_path = ""
    for i, data in enumerate(tqdm(dataset, unit="sample")):
        img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
            continue

        answer = process_sample(model, processor, img_path, data["Question"])
        results.append(
            {
                "ID": data["ID"],
                "Question": data["Question"],
                "Predicted_Answer": answer,
                "Ground_Truth": data["Answer"],
                "Attribute": data["Attribute"],
            }
        )

        if i % 10 == 0:
            intermediate_path = save_path.replace(".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(intermediate_path, "w") as f:
                json.dump(results, f, indent=4, default=str)
            if prev_path:
                os.remove(prev_path)
            logger.info(f"Intermediate results saved to {intermediate_path} and deleted {prev_path}.")
            prev_path = intermediate_path

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Results saved to {save_path}.")


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./data/eval2/QA_Eval2.json", help="Path to dataset")
    parser.add_argument("--image_folder", type=str, default="./data/processed_images", help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--save_path", type=str, default="./results/results_InstructBLIP.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, processor = load_model(args.model_source)
    model.to(device)

    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    evaluate(model, processor, dataset, args.image_folder, args.save_path)
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

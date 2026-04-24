"""HumanI Bench — task 2 (instance identity): Magma 8B."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


# Model paths
MODEL_DIR = "/model-weights/Magma-8B"  # Local model path
HF_MODEL_ID = "microsoft/Magma-8B"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
offload_folder = ""


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load and return the Magma-8B model and processor."""
    print(f"Loading Magma-8B model from {'local directory' if model_source == 'local' else 'Hugging Face'}...")
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
        _attn_implementation="flash_attention_2",
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    return model, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize an image while preserving its aspect ratio."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, processor: Any, img_path: str, question: str, device: Any) -> str:
    """Generate an answer for a single image-question pair."""
    try:
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        convs = [
            {
                "role": "system",
                "content": "Given question, answer in the following format: Answer:<answer> Reasoning:<reasoning> in the context of the image.",
            },
            {"role": "user", "content": f"<image_start><image><image_end>\n{question}"},
        ]
        prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory left: {total_mem - torch.cuda.memory_reserved() / 1e9:.2f} GB")

        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = output[:, inputs["input_ids"].shape[1] :]
        predicted_answer = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return predicted_answer if predicted_answer else "No answer generated"
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str) -> None:
    """Process the dataset and save results to a JSON file."""
    results = []
    logger.info("Starting evaluation...")
    prev_path = ""
    for i, data in enumerate(tqdm(dataset, unit="sample")):
        img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
            continue
        answer = process_sample(model, processor, img_path, data["Question"], next(model.parameters()).device)
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
            intermediate_results_path = save_path.replace(
                ".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(intermediate_results_path, "w") as f:
                json.dump(results, f, indent=4, default=str)
            if prev_path:
                os.remove(prev_path)
            logger.info(f"Intermediate results saved to {intermediate_results_path} and deleted {prev_path}.")
            prev_path = intermediate_results_path
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
        "--save_path", type=str, default="./results/results_Magma.json", help="Output file to save results"
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

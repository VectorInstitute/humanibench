"""HumanI Bench — task 2 (instance identity): CogVLM2 Llama3 Chat 19B."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model paths
MODEL_DIR = "/model-weights/cogvlm2-llama3-chat-19B/"  # Local model path
HF_MODEL_ID = "THUDM/cogvlm2-llama3-chat-19B"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directories (if needed)
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
offload_folder = ""


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load and return the model and tokenizer."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading CogVLM2 model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_type = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, cachedir=os.environ["TRANSFORMERS_CACHE"]
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_type, cache_dir=os.environ["TRANSFORMERS_CACHE"], trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    logger.info("Model loaded successfully.")
    return model, tokenizer


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize an image while maintaining aspect ratio."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, tokenizer: Any, img_path: str, question: str) -> str:
    """Process a single image-question pair and generate an answer."""
    try:
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        user_prompt = (
            f"USER: Given question, answer in the following format:"
            f"Question:{question}"
            f"Answer:<answer> Reasoning:<reasoning> in the context of the image. "
            f"ASSISTANT:"
        )

        history: list[Any] = []
        input_by_model = model.build_conversation_input_ids(
            tokenizer, query=user_prompt, history=history, images=[image], template_version="chat"
        )
        device = next(model.parameters()).device
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(device),
            "images": [[input_by_model["images"][0].to(device).to(torch.bfloat16)]] if image is not None else None,
        }
        gen_kwargs = {"max_new_tokens": 256, "pad_token_id": 128002}
        with torch.no_grad():
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            outputs = model.generate(**inputs, **gen_kwargs, do_sample=False, use_cache=True)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<|end_of_text|>")[0]
        print("\nCogVLM2:", response)
        return response
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(model: Any, tokenizer: Any, dataset: list[Any], image_folder: str, save_path: str) -> None:
    """Evaluate the model on the dataset and save results to a JSON file."""
    results = []
    logger.info("Starting evaluation...")
    prev_path = ""
    for i, data in enumerate(tqdm(dataset, unit="sample")):
        img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
            continue
        answer = process_sample(model, tokenizer, img_path, data["Question"])
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
        "--save_path", type=str, default="./results/results_CogVLM2.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model_source)
    model.to(device)
    if torch.cuda.is_available():
        logger.info(f"Model loaded on GPU {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Model loaded on CPU")
    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    logger.info(f"Loaded dataset with {len(dataset)} samples.")
    evaluate(model, tokenizer, dataset, args.image_folder, args.save_path)
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

"""HumanI Bench — task 3 (multiple-choice VQA): CogVLM2 Llama3 Chat 19B."""

import json
import logging
import os
import re
import time
from argparse import ArgumentParser
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


set_seed(45)

MAX_NEW_TOKENS = 125

# Environment variables for cache and model directories
MODEL_DIR = os.getenv("MODEL_DIR", "/model-weights/cogvlm2-llama3-chat-19B/")  # Default local model path
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "THUDM/cogvlm2-llama3-chat-19B")  # Default Hugging Face Model ID

# Cache and offload folders from environment variables with default values

os.environ["HF_HOME"] = os.getenv("HF_HOME", "")
os.environ["HF_HUB_TMP"] = os.getenv("HF_HUB_TMP", "")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "")


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT = """Answer the question using one of the given choices based on the image.

Question:
{QUESTION}

Provide the response by including the Answer and Reasoning in JSON format:
{{"Answer": "The correct letter and option",
"Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}
"""


def extract_answer_and_reason(text: str) -> tuple[str, str | None]:
    """Extract the chosen option (e.g., 'B. Female') and the reasoning from the VLM response."""
    try:
        # Try extracting JSON content first
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            answer = data.get("Answer", "").strip()
            reasoning = data.get("Reasoning", data.get("Explanation", "")).strip()
            if answer and reasoning:
                return answer, reasoning
    except Exception:
        pass

    # If JSON parsing fails, extract using regex
    try:
        pattern = r'([A-D]\.\s+[^\n]+)\s*(?:\*\*Reasoning:\*\*|Reasoning:|Explanation:)?\s*"?([^"\n]*)"?'

        match = re.search(pattern, text, re.DOTALL)

        if match:
            answer = match.group(1).strip()  # Extracts the full option (e.g., "B. Female")
            reasoning = match.group(2).strip() if match.group(2) else None  # Extracts reasoning
            return answer, reasoning

    except Exception:
        pass

    # If nothing matches, return text as fallback
    return text, None


# Load the model and tokenizer
def load_model(model_source: str = "local") -> tuple[Any, Any]:
    # Select model source
    """Load model."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading CogVLM2 model from {model_path}...")

    # Determine optimal torch precision type
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


# Resize image dynamically
def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize image."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image  # Return the resized PIL Image object
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


# Process a single image-question pair
def process_sample(model: Any, tokenizer: Any, img_path: str, question: str) -> str:
    """Process sample."""
    try:
        # Load & resize the image
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        query_prompt = PROMPT.format(QUESTION=question)

        history: list[Any] = []
        input_by_model = model.build_conversation_input_ids(
            tokenizer, query=query_prompt, history=history, images=[image], template_version="chat"
        )
        device = next(model.parameters()).device

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(device),
            "images": [[input_by_model["images"][0].to(device).to(torch.bfloat16)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": 128002,
        }
        with torch.no_grad():
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            outputs = model.generate(**inputs, **gen_kwargs, do_sample=False, use_cache=True)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(outputs[0])
            print(response)
            response = response.split("<|end_of_text|>")[0]

        print("\nCogVLM2:", response)
        return response

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


# Main function to process dataset
def evaluate(
    model: Any, tokenizer: Any, dataset: list[Any], image_folder: str, save_path: str, mode: str = "single"
) -> None:
    """Evaluate."""
    results = []
    logger.info(f"Starting evaluation in {mode} mode...")
    intermediate_results_path = save_path.replace(".json", "_intermediate.json")
    prev_path = ""
    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")

            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue

            question_with_choices = data["Question"] + "\n" + data["Options"]

            response = process_sample(model, tokenizer, img_path, question_with_choices)
            answer, reason = extract_answer_and_reason(response)

            results.append(
                {
                    "ID": data["ID"],
                    "Question": question_with_choices,
                    "Predicted_Answer": answer,
                    "Predicted_Reasoning": reason,
                    "Ground_Truth_Answer": data["Answer"],
                    "Ground_Truth_Reasoning": data["Reasoning"],
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

    # Command-line arguments with default values
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (JSON file)")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--save_path", type=str, required=True, help="Output file to save results")

    # Optional arguments with default values
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: 'cuda')")
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["local", "hf"],
        default="hf",
        help="Model source: 'local' or 'hf' (default: 'hf')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="single",
        help="Single or batch processing (default: 'single')",
    )

    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, tokenizer = load_model(args.model_source)
    model.to(device)

    # Print the device of model
    if torch.cuda.is_available():
        logger.info(f"Model loaded on GPU {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Model loaded on CPU")

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, tokenizer, dataset, args.image_folder, args.save_path, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

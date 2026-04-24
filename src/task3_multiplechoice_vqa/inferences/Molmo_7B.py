"""HumanI Bench — task 3 (multiple-choice VQA): Molmo 7B."""

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
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, set_seed


# Setting seed for reproducibility
set_seed(45)

# Constants
MAX_NEW_TOKENS = 150
MODEL_DIR = "/model-weights/Molmo-7B-D-0924"
HF_MODEL_ID = "allenai/Molmo-7B-D-0924"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directories
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""

# Prompt template for the task
PROMPT = """Answer the question using one of the given choices based on the image.

Question:
{QUESTION}

Provide the response only in the following JSON format:
{{"Answer": "The correct letter and option",
"Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

Do not provide any other extra information.
"""


def extract_answer_and_reason(text: str) -> tuple[str, str | None]:
    """Extract the answer and reason from the VLM response."""
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            answer = data.get("Answer", "").strip()
            reasoning = data.get("Reasoning", "").strip()
            if answer and reasoning:
                return answer, reasoning
    except Exception:
        pass

    # If not JSON, use regex to extract
    pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'
    match = re.search(pattern, text)

    if match:
        answer = match.group(1)
        reasoning = match.group(2)
        return answer, reasoning
    return text, None


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load the model and processor from the given source (local or Hugging Face Hub)."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading Molmo 7B model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        torch_dtype="auto",
        device_map=device,
    ).eval()

    # Load the processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        torch_dtype="auto",
        device_map=device,
    )

    return model, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resizes the image to a maximum size."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, processor: Any, img_path: str, question: str) -> str:
    """Process one image-question pair and return an answer."""
    try:
        # Resize image
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        # Format the prompt with the question
        question_prompt = PROMPT.format(QUESTION=question)

        # Process the image and text
        inputs = processor.process(
            images=[image],
            text=question_prompt,
        )

        # Move inputs to the correct device
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # Generate output
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer,
            )

        # Decode the generated tokens to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def save_intermediate_results(results: list[Any], save_path: str, i: int) -> str:
    """Save intermediate results every 10 samples."""
    intermediate_results_path = save_path.replace(".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(intermediate_results_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Intermediate results saved to {intermediate_results_path}")
    return intermediate_results_path


def evaluate(
    model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str, mode: str = "single"
) -> None:
    """Evaluate the model on a dataset and save results."""
    results = []
    logger.info(f"Starting evaluation in {mode} mode...")
    prev_path = None

    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue

            question_with_choices = data["Question"] + "\n" + data["Options"]
            response = process_sample(model, processor, img_path, question_with_choices)
            answer, reason = extract_answer_and_reason(response)

            results.append(
                {
                    "ID": data["ID"],
                    "Question": data["Question"],
                    "Predicted_Answer": answer,
                    "Predicted_Reasoning": reason,
                    "Ground_Truth_Answer": data["Answer"],
                    "Ground_Truth_Reasoning": data["Reasoning"],
                    "Attribute": data["Attribute"],
                }
            )

            # Save intermediate results every 10 samples
            if i % 10 == 0:
                intermediate_results_path = save_intermediate_results(results, save_path, i)

                # Delete the previous intermediate file
                if prev_path and os.path.exists(prev_path):
                    os.remove(prev_path)
                    logger.info(f"Deleted previous intermediate results: {prev_path}")

                prev_path = intermediate_results_path

            pbar.update(1)

    # Save final results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    logger.info(f"Results saved to {save_path}.")


# Main function
if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--image_folder", type=str, help="Path to image folder")
    parser.add_argument("--device", type=str, help="Device to run the model on")
    parser.add_argument("--save_path", type=str, help="Output file to save results")
    parser.add_argument("--model_source", type=str, help="Model source: 'local' or 'hf'")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], help="Single or batch processing")

    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, processor = load_model(args.model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

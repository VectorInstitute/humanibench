"""HumanI Bench — task 3 (multiple-choice VQA): Instructblip Vicuna 7B."""

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
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor


MAX_TOKENS = 120

# Default model directory and Hugging Face Model ID
MODEL_DIR = "/model-weights/instructblip-vicuna-7b/"
HF_MODEL_ID = "Salesforce/instructblip-vicuna-7b"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Find the first '{' and the last '}' to extract a JSON string.
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

    # If not JSON then extract the text directly using regex
    try:
        pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'

        match = re.search(pattern, text)

        if match:
            answer = match.group(1)
            reasoning = match.group(2)
            return answer, reasoning
        return text, None
    except Exception:
        return text, None


# Load the model and processor
def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load model."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(
        f"Loading InstructBLIP model from {'local directory' if model_source == 'local' else 'Hugging Face'}..."
    )

    model = InstructBlipForConditionalGeneration.from_pretrained(model_path).eval()
    processor = InstructBlipProcessor.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, processor


# Resize image
def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize image."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)  # Resize in place while maintaining aspect ratio
        return image  # Return the resized PIL image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None  # Return None if resizing fails


# Process a single image-question pair and generate an answer
def process_sample(model: Any, processor: Any, img_path: str, question: str) -> str:
    """Process sample."""
    try:
        device = next(model.parameters()).device  # Get the device of the model
        logger.info(f"Processing {img_path} with question: {question}...")
        logger.info(f"Device: {device}")

        # Resize image but don't save
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        query_prompt = PROMPT.format(QUESTION=question)

        inputs = processor(images=image, text=query_prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_TOKENS,
            min_length=len(query_prompt),
            repetition_penalty=1.5,
            length_penalty=0.8,
            num_beams=4,
            temperature=0.5,
            max_length=512,
        )
        input_len = inputs["input_ids"].shape[1]
        outputs = outputs[:, input_len:]
        return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


# Main function to process dataset
def evaluate(
    model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str, mode: str = "single"
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

            response = process_sample(model, processor, img_path, question_with_choices)
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

    # Command-line arguments
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
    model, processor = load_model(args.model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

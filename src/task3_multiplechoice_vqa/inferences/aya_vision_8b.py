"""HumanI Bench — task 3 (multiple-choice VQA): aya vision 8b."""

import base64
import json
import logging
import os
import re
import time
from argparse import ArgumentParser
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed


# Set a random seed for reproducibility
set_seed(45)

# Constants
MAX_NEW_TOKENS = 150
PROMPT = """Answer the question using one of the given choices based on the image.

Question:
{QUESTION}

Provide the response only in the following JSON format:
{{"Answer": "The correct letter and option",
"Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

Do not provide any other extra information.
"""

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variable paths (set these in your environment or .env file)
MODEL_DIR = os.getenv("MODEL_DIR", "/model-weights/aya-vision-8b")  # Local model path
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "CohereForAI/aya-vision-8b")  # Hugging Face Model ID

# Offload and cache directories to be set in the environment
OFFLOAD_FOLDER = os.getenv("OFFLOAD_FOLDER", "")
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "")
offload_folder = ""


# Resize image function
def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize image."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image  # Returns the resized PIL image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None  # Return None if resizing fails


# Extract answer and reasoning from the model's response
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
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")

    # Fallback to regex extraction if JSON fails
    try:
        pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'
        match = re.search(pattern, text)

        if match:
            answer = match.group(1)
            reasoning = match.group(2)
            return answer, reasoning
        return text, None
    except Exception as e:
        logger.error(f"Error with regex extraction: {e}")
        return text, None


# Load the model and processor from local or Hugging Face
def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load the model and processor for the given source ('local' or 'hf')."""
    logger.info(
        f"Loading Magma-8B Vision model from {'local directory' if model_source == 'local' else 'Hugging Face'}..."
    )

    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, device_map="auto", offload_folder=OFFLOAD_FOLDER, trust_remote_code=True, cache_dir=CACHE_DIR
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir=CACHE_DIR)

    return model, processor


# Process a single image-question pair and generate an answer
def process_sample(model: Any, processor: Any, img_path: str, question: str, device: Any) -> str:
    """Process sample."""
    try:
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        query_prompt = PROMPT.format(QUESTION=question)

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
                    {"type": "text", "text": query_prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
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

            response = process_sample(model, processor, img_path, question_with_choices, device)
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

            # Save intermediate results every 10 samples
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

    # Save final results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    logger.info(f"Results saved to {save_path}.")


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (JSON file)")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--save_path", type=str, required=True, help="Output file to save results")
    parser.add_argument(
        "--model_source", type=str, choices=["local", "hf"], default="hf", help="Model source: 'local' or 'hf'"
    )
    parser.add_argument(
        "--mode", type=str, choices=["single", "batch"], default="single", help="Single or batch processing"
    )

    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, processor = load_model(args.model_source)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

# To run this script, use the following command:
# python aya_vision_8b.py \
#     --dataset <path_to_dataset_json> \
#     --image_folder <path_to_image_folder> \
#     --device <cuda_or_cpu> \
#     --save_path <path_to_save_results_json> \
#     --model_source <local_or_hf> \
#     --mode <single_or_batch>

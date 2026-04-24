"""HumanI Bench — task 3 (multiple-choice VQA): Llama 3 2 11B Vision Instruct."""

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
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, set_seed


set_seed(45)

# Default Model paths (in case they are not passed)
MODEL_DIR = "/model-weights/Llama-3.2-11B-Vision-Instruct/"  # Local model path
HF_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # Hugging Face Model ID

# Parameters
MAX_NEW_TOKENS = 120
OFFLOAD_FOLDER = ""  # Offload folder for model weights

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
    pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'

    match = re.search(pattern, text)

    if match:
        answer = match.group(1)
        reasoning = match.group(2)
        return answer, reasoning
    return text, None


# Load the model and processor
def load_model(model_source: str, quantized: bool = False) -> tuple[Any, Any]:
    """Load model."""
    logger.info(f"Loading Llama 3.2 Vision model from {'local' if model_source == 'local' else 'Hugging Face'}...")

    # Select model path based on source (local or Hugging Face)
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)

    # 4-bit quantization settings
    if quantized:
        logger.info("Using 4-bit quantization for efficient memory usage.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto",
            offload_folder=OFFLOAD_FOLDER,
        )
    else:
        logger.info("Using full-precision model (FP16).")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", offload_folder=OFFLOAD_FOLDER
        )

    logger.info("Model loaded successfully.")
    return model, processor


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


# Process a single image-question pair and generate an answer
def process_sample(model: Any, processor: Any, img_path: str, question: str, device: Any) -> str:
    """Process sample."""
    try:
        # Resize image before processing
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        question_prompt = PROMPT.format(QUESTION=question)

        # Construct conversation prompt
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_prompt}]}]

        # Apply chat template
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Prepare inputs
        inputs = processor(images=image, text=input_text, add_special_tokens=False, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        # Decode the output
        predicted_answer = processor.decode(output[0], skip_special_tokens=True)
        predicted_answer = predicted_answer[
            predicted_answer.find("assistant") + 9 :
        ]  # To extract only the assistant response

        return predicted_answer if predicted_answer else "No answer generated"

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


# Main function to process dataset
def evaluate(model: Any, processor: Any, dataset: list[Any], image_folder: str, device: Any, save_path: str) -> None:
    """Evaluate."""
    results = []
    logger.info("Starting evaluation...")

    with tqdm(total=len(dataset), unit="sample") as pbar:
        for data in dataset:
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
            question_with_choices = data["Question"] + "\n" + data["Options"]

            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue  # Skip missing images

            response = process_sample(model, processor, img_path, question_with_choices, device)
            answer, reason = extract_answer_and_reason(response)
            results.append(
                {
                    "ID": data["ID"],
                    "Question": question_with_choices,
                    "Predicted_Answer": answer,
                    "Predicted_Reasoning": reason,
                    "Ground_Truth_Answer": data.get("Answer", "N/A"),
                    "Ground_Truth_Reasoning": data.get("Reasoning", "N/A"),
                    "Attribute": data.get("Attribute", "N/A"),
                }
            )

            pbar.update(1)

    # Save results to JSON
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {save_path}.")


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--image_folder", type=str, help="Path to image folder")
    parser.add_argument("--device", type=str, help="Device to run the model on")
    parser.add_argument("--save_path", type=str, help="Output file to save results")
    parser.add_argument("--model_source", type=str, help="Model source: 'local' or 'hf'")

    args = parser.parse_args()

    # Check if all required arguments are provided
    if not all([args.dataset, args.image_folder, args.device, args.save_path, args.model_source]):
        raise ValueError("All arguments are required: --dataset, --image_folder, --device, --save_path, --model_source")

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model (using the model_source argument)
    model, processor = load_model(args.model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, device, args.save_path)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

"""HumanI Bench — task 3 (multiple-choice VQA): Phi35 Vision Instruct."""

import json
import logging
import os
import re
import time
from argparse import ArgumentParser
from typing import Any

import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, set_seed


# Load environment variables from .env file
load_dotenv()

set_seed(45)

# Model directory
MODEL_DIR = "/model-weights/Phi-3.5-vision-instruct"  # Local model path
HF_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"  # Hugging Face Model ID

# Parameters
MAX_NEW_TOKENS = 200

# Cache directory taken from .env
HF_HOME = os.getenv("HF_HOME")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT = """Answer the question using one of the given choices based on the image.

Question:
{QUESTION}

Provide the response only in the following JSON format:
{{"Answer": "Must contain the correct letter along with corresponding text",
"Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

Always provide option including its corresponding text in "Answer". Do not provide any other extra information.
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
def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load model."""
    print(
        f"Loading Phi-3-Vision-128K-Instruct model from {'local directory' if model_source == 'local' else 'Hugging Face'}..."
    )

    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Required for Phi-3
        torch_dtype="auto",  # Automatically selects best precision (FP16/BF16)
        device_map="auto",  # Automatically assigns to GPU
        _attn_implementation="flash_attention_2",  # Optimized for fast inference
        # _attn_implementation='eager'  # Default implementation
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


# Resize image
def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize image."""
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

        question_prompt = PROMPT.format(QUESTION=question)
        # Phi-3.5 Vision expects chat format with special tokens
        prompt = f"<|user|>\n<|image_1|>\n{question_prompt}\n<|end|>\n<|assistant|>\n"

        # Prepare inputs
        inputs = processor(prompt, [image], return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                do_sample=False,
            )

        # Remove input tokens from output & decode response
        generated_ids = output[:, inputs["input_ids"].shape[1] :]
        predicted_answer = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

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
                continue

            response = process_sample(model, processor, img_path, question_with_choices, device)
            answer, reason = extract_answer_and_reason(response)

            results.append(
                {
                    "ID": data["ID"],
                    "Question": question_with_choices,
                    "Predicted_Answer": answer,
                    "Predicted_Reasoning": reason,
                    "Ground_Truth": data.get("Answer", "N/A"),
                    "Ground_Truth_Reasoning": data.get("Reasoning", "N/A"),
                    "Attribute": data.get("Attribute", "N/A"),
                }
            )

            pbar.update(1)

    # Save results to JSON
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    logger.info(f"Results saved to {save_path}.")


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments without defaults
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset JSON file")
    parser.add_argument("--image_folder", type=str, help="Path to folder containing images")
    parser.add_argument("--device", type=str, help="Device to run the model on")
    parser.add_argument("--save_path", type=str, help="Output file to save results")
    parser.add_argument("--model_source", type=str, help="Model source: 'local' or 'hf'")

    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Model source
    model_source = "huggingface" if args.model_source == "hf" else "local"
    logger.info(f"Model source: {model_source}")

    # Load model
    model, processor = load_model(model_source)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Select first 50 samples for testing
    # dataset = dataset[:20]
    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, device, args.save_path)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

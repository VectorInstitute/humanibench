"""HumanI Bench — task 3 (multiple-choice VQA): Llava v1 6 7B."""

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
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, set_seed


# Set a random seed for reproducibility
set_seed(45)

# Constants
MAX_NEW_TOKENS = 200
MODEL_DIR = "/model-weights/llava-v1.6-vicuna-7b-hf"
HF_MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
CACHE_DIR = ""

# Environment variables for caching
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt template for the VLM model
PROMPT = """Answer the question using one of the given choices based on the image.

Question:
{QUESTION}

Provide the response only in the following JSON format:
{{"Answer": "Must contain the correct letter along with corresponding text",
"Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

Do not provide any other extra information.
"""


def extract_answer_and_reason(text: str) -> tuple[str, str | None]:
    """Extract the answer and reasoning from the VLM response."""
    try:
        # Attempt to parse JSON formatted answer
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            return data.get("Answer", "").strip(), data.get("Reasoning", "").strip()
    except Exception:
        pass

    # If JSON parsing fails, attempt regex-based extraction
    pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'
    match = re.search(pattern, text)
    if match:
        return match.group(1), match.group(2)
    return text, None


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load the Llava model and processor."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading Llava model from {model_path}...")

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        cache_dir=CACHE_DIR,
    ).to("cuda:0")

    processor = LlavaNextProcessor.from_pretrained(model_path, cache_dir=CACHE_DIR)
    return model, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize the input image to fit the max size."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, processor: Any, img_path: str, question: str) -> str:
    """Process a single image-question pair and generate an answer."""
    try:
        # Load & resize the image
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        query_prompt = PROMPT.format(QUESTION=question)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query_prompt},
                    {"type": "image"},
                ],
            },
        ]

        # Prepare inputs for the model
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # Generate output from the model
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        answer = processor.decode(output[0], skip_special_tokens=True)

        # Extract the answer after "ASSISTANT:"
        return answer[answer.find("ASSISTANT:") :]

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def save_intermediate_results(results: list[Any], save_path: str, i: int) -> str:
    """Save intermediate results every 50 samples."""
    intermediate_results_path = save_path.replace(".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(intermediate_results_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Intermediate results saved to {intermediate_results_path}")
    return intermediate_results_path


def evaluate(
    model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str, mode: str = "single"
) -> None:
    """Evaluate the model on a dataset."""
    results = []
    logger.info(f"Starting evaluation in {mode} mode...")

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

            # Save intermediate results every 10 samples
            if i % 10 == 0:
                intermediate_results_path = save_intermediate_results(results, save_path, i)
                if prev_path:
                    os.remove(prev_path)
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
    parser.add_argument("--dataset", type=str, default="./data/eval3/QA_Eval3.json", help="Path to dataset")
    parser.add_argument("--image_folder", type=str, default="./data/processed_images", help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--save_path", type=str, default="./results/results_Llava_v1_6_7B.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="hf", help="Model source: 'local' or 'hf'")
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "batch"], help="Single or batch processing"
    )

    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, processor = load_model(args.model_source)
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
    evaluate(model, processor, dataset, args.image_folder, args.save_path, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

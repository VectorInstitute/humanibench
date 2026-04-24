"""HumanI Bench — task 3 (multiple-choice VQA): Qwen25 VL 7B Instruct."""

import json
import logging
import os
import re
import time
from argparse import ArgumentParser
from typing import Any

import torch

# Load environment variables from .env file
from dotenv import load_dotenv
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed


load_dotenv()

set_seed(45)

MAX_NEW_TOKENS = 120

# Model directory
MODEL_DIR = "/model-weights/Qwen2.5-VL-7B-Instruct/"
HF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache Directories taken from .env
HF_HOME = os.getenv("HF_HOME")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")

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

    # If not JSON, extract the text directly using regex
    try:
        pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            reasoning = match.group(2)
            return answer, reasoning
    except Exception:
        pass

    return text, None


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load model."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading Qwen2.5 model from {model_path}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, cache_dir=TRANSFORMERS_CACHE, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", revision="refs/pr/24", cache_dir=TRANSFORMERS_CACHE
    )

    return model, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resizes the image to fit within the specified max size."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, processor: Any, img_path: str, question: str) -> str:
    """Process an image-question pair and generate an answer."""
    try:
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        question_prompt = PROMPT.format(QUESTION=question)

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question_prompt}]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str) -> None:
    """Evaluate the model on the dataset and save the results."""
    results = []
    logger.info("Starting evaluation...")

    intermediate_results_path = save_path.replace(".json", "_intermediate.json")
    prev_path = ""

    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            if data["ID"] in ["a8e117d00e", "05b85dedea"]:
                img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
                question_with_choices = data["Question"] + "\n" + data["Options"]

                if not os.path.exists(img_path):
                    logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                    continue

                response = process_sample(model, processor, img_path, question_with_choices)
                answer, reason = extract_answer_and_reason(response[0])
                results.append(
                    {
                        "ID": data["ID"],
                        "Question": question_with_choices,
                        "Predicted_Answer": answer,
                        "Predicted_Reasoning": reason,
                        "Ground_Truth": data["Answer"],
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

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, processor = load_model(args.model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Select 20 samples for testing
    # dataset = dataset[:20]
    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

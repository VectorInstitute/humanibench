"""HumanI Bench — task 3 (multiple-choice VQA): JanusPro."""

import json
import logging
import os
import re
import time
from argparse import ArgumentParser
from typing import Any

import torch
from janus.models import VLChatProcessor
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, set_seed


set_seed(45)

# User Keys
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""

# Parameters
MAX_NEW_TOKENS = 125
TEMPERATURE = 0.7  # Sampling temperature (for batch processing)
DO_SAMPLE = False  # Set to False to generate deterministic output

# Model directory
MODEL_DIR = "/model-weights/Janus-Pro-7B/"  # Load the model from model-weights
HF_MODEL_ID = "deepseek-ai/Janus-Pro-7B"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT = """Answer the question using one of the given choices based on the image.

Question:
{QUESTION}

Provide the response only in the following JSON format:
{{"Answer": "The correct letter and option",
"Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

Do not provide any other extra information. Strictly include the correct letter and corresponding text in answer.
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


def load_model(model_source: str = "local") -> tuple[Any, Any, Any]:
    """Load model."""
    print(f"Loading Janus-Pro model from {'local directory' if model_source == 'local' else 'Hugging Face'}...")

    # Ensure the directory exists
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    # Load processor
    processor = VLChatProcessor.from_pretrained(model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    tokenizer = processor.tokenizer

    # Load model
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],  # Ensure download happens in scratch space
            trust_remote_code=True,
        )
        .to(torch.bfloat16)
        .cuda()
        .eval()
    )

    return model, tokenizer, processor


# Resize image
def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize image."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


# Process a single image-question pair and generate an answer
def process_sample(model: Any, tokenizer: Any, processor: Any, img_path: str, question: str, device: Any) -> str:
    """Process sample."""
    try:
        # Load & resize the image
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        # Ensure image is passed as a list (Janus expects a list of images)
        pil_images = [image]
        question_prompt = PROMPT.format(QUESTION=question)

        # Construct conversation
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{question_prompt}", "images": pil_images},
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Preprocess input
        prepared_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)

        # Generate response
        inputs_embeds = model.prepare_inputs_embeds(**prepared_inputs)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

        # Decode response
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer if answer else "No answer generated"

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


# Batch processing
def process_batch(
    model: Any, tokenizer: Any, processor: Any, dataset: list[Any], image_folder: str, device: Any
) -> list[Any]:
    """Process batch."""
    try:
        batch_conversations = []
        image_list = []  # Store PIL images

        for data in dataset:
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")

            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue

            # Load and resize image
            image = resize_image(img_path)
            if image is None:
                continue

            question_prompt = PROMPT.format(QUESTION=data["Question"] + "\n" + data["Options"])

            # Ensure image is added as a list
            batch_conversations.append(
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question_prompt}",
                    "images": [image],  # Wrap image in a list
                }
            )
            image_list.append(image)

        if not batch_conversations:
            return []

        # Ensure processor receives a list of images
        prepared_inputs = processor(conversations=batch_conversations, images=image_list, force_batchify=True).to(
            model.device
        )

        # Generate responses
        inputs_embeds = model.prepare_inputs_embeds(**prepared_inputs)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            use_cache=False,
        )

        # Decode batch responses
        return tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return []


# Main function to process dataset
def evaluate(
    model: Any,
    tokenizer: Any,
    processor: Any,
    dataset: list[Any],
    image_folder: str,
    device: Any,
    save_path: str,
    mode: str = "single",
) -> None:
    """Evaluate."""
    results = []
    logger.info(f"Starting evaluation in {mode} mode...")

    with tqdm(total=len(dataset), unit="sample") as pbar:
        if mode == "batch":
            responses = process_batch(model, tokenizer, processor, dataset, image_folder, device)
            for idx, data in enumerate(dataset):
                answer: str | None
                reason: str | None
                if idx < len(responses):
                    answer, reason = extract_answer_and_reason(str(responses[idx]))
                else:
                    answer, reason = None, None
                results.append(
                    {
                        "ID": data["ID"],
                        "Question": data["Question"] + "\n" + data["Options"],
                        "Predicted_Answer": answer,
                        "Predicted_Reasoning": reason,
                        "Ground_Truth_Answer": data.get("Answer", "N/A"),
                        "Ground_Truth_Reasoning": data.get("Reason", "N/A"),
                        "Attribute": data.get("Attribute", "N/A"),
                    }
                )
                pbar.update(1)

        else:  # Single processing
            for data in dataset:
                img_path = os.path.join(image_folder, f"{data['ID']}.jpg")

                if not os.path.exists(img_path):
                    logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                    continue

                question_with_choices = data["Question"] + "\n" + data["Options"]
                response = process_sample(model, tokenizer, processor, img_path, question_with_choices, device)
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

    # Save results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

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
    parser.add_argument("--model_source", type=str, choices=["local", "hf"], help="Model source: 'local' or 'hf'")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], help="Single or batch processing")

    args = parser.parse_args()

    # Check if all required arguments are provided
    if not all([args.dataset, args.image_folder, args.device, args.save_path, args.model_source, args.mode]):
        raise ValueError(
            "All arguments are required: --dataset, --image_folder, --device, --save_path, --model_source, --mode"
        )

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, tokenizer, processor = load_model(args.model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, tokenizer, processor, dataset, args.image_folder, device, args.save_path, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

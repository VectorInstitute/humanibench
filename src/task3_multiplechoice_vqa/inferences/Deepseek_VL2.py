"""HumanI Bench — task 3 (multiple-choice VQA): Deepseek VL2."""

import json
import logging
import os
import re
import time
from argparse import ArgumentParser
from typing import Any

import torch
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from PIL import Image
from tqdm import tqdm


MAX_NEW_TOKENS = 150

# Cache Directory Setup
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""

# Model Directory (Hugging Face Model ID)
HF_MODEL_ID = "deepseek-ai/deepseek-vl2-small"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt template
PROMPT = """Answer the question using one of the given choices based on the image.

Question:
{QUESTION}

Provide the response only in the following JSON format:
{{"Answer": "The correct letter along with corresponding text",
"Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

Ensure to provide both option and its corresponding text in answer. Do not provide any other extra information.
"""


def extract_answer_and_reason(text: str) -> tuple[str, str | None]:
    """Extract the answer and reason from the VLM response."""
    try:
        # Attempt to extract JSON from the response
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

    # If not JSON, use regex to extract answer and reasoning
    pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'

    match = re.search(pattern, text)

    if match:
        answer = match.group(1)
        reasoning = match.group(2)
        return answer, reasoning
    return text, None


def load_model(model_source: str = "local") -> tuple[Any, Any, Any]:
    """Load the model and tokenizer."""
    model_path = HF_MODEL_ID  # Using Hugging Face model ID directly
    logger.info(f"Loading Deepseek VL2 model from {model_path}...")

    processor = DeepseekVLV2Processor.from_pretrained(model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    tokenizer = processor.tokenizer  # Tokenizer is part of the processor
    model = (
        DeepseekVLV2ForCausalLM.from_pretrained(
            model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"], trust_remote_code=True
        )
        .to(torch.bfloat16)
        .cuda()
        .eval()
    )

    return model, processor, tokenizer


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize the image to fit within the max_size."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, tokenizer: Any, processor: Any, img_path: str, question: str) -> str:
    """Process a single image-question pair and generate an answer."""
    try:
        # Load and resize the image
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        question_prompt = PROMPT.format(QUESTION=question)

        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{question_prompt}", "images": [image]},
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Preprocess input
        prepared_inputs = processor(conversations=conversation, images=[image], force_batchify=True).to(model.device)

        # Generate response
        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**prepared_inputs)
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepared_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(
    model: Any,
    tokenizer: Any,
    processor: Any,
    dataset: list[Any],
    image_folder: str,
    save_path: str,
    mode: str = "single",
) -> None:
    """Run evaluation on the dataset and write results to disk."""
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

            response = process_sample(model, tokenizer, processor, img_path, question_with_choices)
            answer, reason = extract_answer_and_reason(response)

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

    # Save final results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    logger.info(f"Results saved to {save_path}.")


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments with required parameters
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
    model, processor, tokenizer = load_model(args.model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, tokenizer, processor, dataset, args.image_folder, args.save_path, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

"""HumanI Bench — task 7 (image resilience): Phi4."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


# Model directory
MODEL_DIR = "/model-weights/Phi-4-multimodal-instruct"  # Local model path
HF_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""


# Load the model and processor
def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load model."""
    print(f"Loading Phi-4 Vision model from {'local directory' if model_source == 'local' else 'Hugging Face'}...")

    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Required for Phi-4
        torch_dtype="auto",  # Automatically selects best precision (FP16/BF16)
        device_map="auto",  # Automatically assigns to GPU
        # _attn_implementation='flash_attention_2'  # Optimized for fast inference
        _attn_implementation="eager",  # Default implementation
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

        user_prompt = f"Given question, answer in the following format:\
                Question:{question}\
                Answer:<answer> \
                Reasoning:<reasoning> in the context of the image."

        # Phi-3 Vision expects chat format with special tokens
        prompt = f"<|user|><|image_1|>\n{user_prompt}<|end|><|assistant|>"
        # <|user|><|image_1|>Describe the image in detail.<|end|><|assistant|>
        # Prepare inputs
        inputs = processor(prompt, image, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                use_cache=True,
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
def evaluate(
    model: Any,
    processor: Any,
    dataset: list[Any],
    image_folder: str,
    save_path: str,
    attack: str,
) -> None:
    """Evaluate."""
    results = []
    logger.info("Starting evaluation...")
    intermediate_results_path = save_path.replace(".json", "_intermediate.json")
    prev_path = ""
    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            # print(data)
            img_path = os.path.join(image_folder, f"{data['ID']}_{attack}.jpg")

            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue

            answer = process_sample(model, processor, img_path, data["Question"], device)
            results.append(
                {
                    "ID": data["ID"],
                    "Question": data["Question"],
                    "Predicted_Answer": answer,
                    "Ground_Truth": data["Answer"],
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
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--image_folder", type=str, help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--save_path", type=str, default="./results/results_Phi4.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="hf", help="Model source: 'local' or 'hf'")
    parser.add_argument("--attack", type=str, default="compression", help="Attack type")

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
    # dataset = {k: dataset[k] for k in list(dataset.keys())[:20]}

    logger.info(f"Attack type: {args.attack}")

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, args.attack)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

# To run the script:
# python Phi4.py --dataset <path_to_dataset> \
# --image_folder <path_to_image_folder> \
# --save_path results_Phi4.json \
# --model_source local \
# --attack <attack_type>

"""Llama 3.2 Vision inference."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, set_seed


set_seed(45)

# User Keys
OFFLOAD_FOLDER = ""
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""


# Parameters
MAX_NEW_TOKENS = 256

# Model paths
MODEL_DIR = "/model-weights/Llama-3.2-11B-Vision-Instruct/"  # Local model path
HF_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load the model and processor
def load_model(model_source: str = "local", quantized: bool = False) -> tuple[Any, Any]:
    """Load model."""
    logger.info(f"Loading Llama 3.2 Vision model from {'local' if model_source == 'local' else 'Hugging Face'}...")

    # Select source path
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )

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
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
        )
    else:
        logger.info("Using full-precision model (FP16).")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=OFFLOAD_FOLDER,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
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

        user_prompt = f"Given question, answer in the following format:\
                Question:{question}\
                Answer:<answer> \
                Reasoning:<reasoning> in the context of the image."

        # question_prompt = PROMPT.format(QUESTION=question)
        question_prompt = user_prompt

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
    # count = 0
    intermediate_results_path = save_path.replace(".json", "_intermediate.json")
    prev_path = ""
    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            img_path = os.path.join(image_folder, f"{data['ID']}_{attack}.jpg")

            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue  # Skip missing images

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

    # Save results to JSON
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
        "--save_path", type=str, default="./results/results_Llama.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="hf", help="Model source: 'local' or 'hf'")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to process")
    parser.add_argument("--quantized", type=bool, default=False, help="Use quantized model")
    parser.add_argument("--attack", type=str, default="compression", help="Attack type")

    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine model source (local vs Hugging Face)
    model_source = "huggingface" if args.model_source == "hf" else "local"
    logger.info(f"Model source: {model_source}")

    # Load model
    model, processor = load_model(model_source, quantized=args.quantized)
    model.to(device)

    # In the dataset path, after the last _ and before .json is the language
    language = args.dataset.split("_")[-1].split(".")[0]
    logger.info(f"Processing dataset in {language} language with {args.mode} mode...")

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Adjust sample count if needed
    # dataset = dataset[:20]

    if args.num_samples > 0:
        dataset = dataset[args.num_samples :]

    # print the attack
    logger.info(f"Attack type: {args.attack}")

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, args.attack)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
    logger.info("Evaluation completed.")


# To run the script:
# python Llama.py --dataset <path_to_dataset> \
# --image_folder <path_to_image_folder> \
# --save_path <path_to_save_results> \
# --attack <attack_type>

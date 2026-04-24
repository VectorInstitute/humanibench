"""HumanI Bench — task 4 (multilingual): aya vision 8b eval2."""

import base64
import json
import logging
import os
import time
from argparse import ArgumentParser
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# Model directory and Hugging Face Model ID
MODEL_DIR = "/model-weights/aya-vision-8b"  # Local model path
HF_MODEL_ID = "CohereForAI/aya-vision-8b"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directories
OFFLOAD_FOLDER = ""  # Path to offload folder
os.environ["HF_HOME"] = ""  # Path where you want to store the huggingface cache
os.environ["TRANSFORMERS_CACHE"] = ""  # Path where you want to store the transformers cache


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """
    Load the Aya-Vision model and its processor.

    Args:
        model_source (str): 'local' to load from a local directory; otherwise load from Hugging Face.

    Returns
    -------
        tuple: (model, processor)
    """
    print(f"Loading Aya-Vision Vision model from {'local directory' if model_source == 'local' else 'Hugging Face'}...")
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        offload_folder=OFFLOAD_FOLDER,
        trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )

    return model, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """
    Resize an image to fit within max_size while preserving its aspect ratio.

    Args:
        img_path (str): Path to the image file.
        max_size (tuple): Maximum (width, height).

    Returns
    -------
        PIL.Image.Image or None: Resized image if successful, otherwise None.
    """
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, processor: Any, img_path: str, question: str, language: str, device: Any) -> str:
    """
    Process a single image-question pair and generate an answer.

    Args:
        model: The loaded model.
        processor: The associated processor.
        img_path (str): Path to the image file.
        question (str): The question to be answered.
        language (str): Language in which the answer should be generated.
        device: Torch device to run the model on.

    Returns
    -------
        str: The generated answer or an error message.
    """
    try:
        # Resize the image
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        # Construct the prompt (using a fixed format)
        user_prompt = (
            f"Given question, answer given in {language} language  in the following format :"
            f"Question:{question}"
            f"Answer:<answer> "
            f"Reasoning:<reasoning> in the context of the image in {language} language."
        )

        # Convert the image to a byte stream and encode as base64
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()
        base64_image_url = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        # Construct messages for the chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_image_url}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        predicted_answer = processor.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return predicted_answer if predicted_answer else "No answer generated"

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str, language: str) -> None:
    """
    Evaluate the model on a dataset of image-question pairs and save the results.

    Args:
        model: The loaded model.
        processor: The associated processor.
        dataset (list): List of data samples.
        image_folder (str): Directory containing images.
        save_path (str): Path to save the final results JSON.
        language (str): Language of the questions and answers.

    Returns
    -------
        None
    """
    results = []
    logger.info("Starting evaluation...")
    prev_path = ""

    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                pbar.update(1)
                continue

            # Construct keys for question and answer based on language
            q_id = f"Question({language})"
            a_id = f"Answer({language})"
            answer = process_sample(model, processor, img_path, data[q_id], language, device)
            results.append(
                {
                    "ID": data["ID"],
                    "Question": data[q_id],
                    "Predicted_Answer": answer,
                    "Ground_Truth": data[a_id],
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
                if prev_path:
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
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--save_path", type=str, help="Output file to save results")
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to process")
    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine language from dataset filename (assumes language is after the last '_' and before '.json')
    language = args.dataset.split("_")[-1].split(".")[0]
    logger.info(f"Processing dataset in {language} language...")

    # Load model and processor
    model, processor = load_model(args.model_source)
    # model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Optionally adjust the number of samples
    if args.num_samples > 0:
        dataset = dataset[args.num_samples :]
    # dataset = dataset[:50]

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, language)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

# To run the script:
# python aya_vision_8b_eval2.py \
# --dataset <path_to_dataset_json> \
# --image_folder <path_to_image_folder> \
# --device <cuda_or_cpu> \
# --save_path <path_to_save_results_json> \
# --model_source <local_or_hf> \
# --num_samples <number_of_samples_to_process>

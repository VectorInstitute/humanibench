"""HumanI Bench — task 4 (multilingual): gemma3 12b eval2."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, set_seed


# Set a seed for reproducibility
set_seed(45)

# User settings and environment variables
OFFLOAD_FOLDER = ""  # Path to offload folder
os.environ["HF_HOME"] = ""  # Path where you want to store the huggingface cache
os.environ["TRANSFORMERS_CACHE"] = ""  # Path where you want to store the transformers cache


# Parameters
MAX_NEW_TOKENS = 256

# Model paths
HF_MODEL_ID = "google/gemma-3-12b-it"
MODEL_DIR = "/model-weights/gemma-3-12b-it"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """
    Load the Gemma3 model from Hugging Face model hub or local path.

    Args:
        model_source (str): Source of the model: 'local' or 'hf'

    Returns
    -------
        model (Gemma3ForConditionalGeneration): The loaded model
        processor (AutoProcessor): The processor for the model
    """
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading Paligemma model from {model_path}...")

    # load the model
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        torch_dtype=torch.bfloat16,
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    return model, processor


# Resize image
def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """
    Resize an image to fit within max_size while preserving its aspect ratio.

    Args:
        img_path (str): Path to the image file.
        max_size (tuple): Maximum (width, height).

    Returns
    -------
        PIL.Image.Image: Resized image if successful, None otherwise.
    """
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


# Process a single image-question pair and generate an answer
def process_sample(model: Any, processor: Any, img_path: str, question: str, language: str) -> str:
    """
    Process a single image-question pair and generate an answer.

    Args:
        model: The loaded language model.
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
        # Resize image before processing
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Construct user prompt
        user_prompt = f"Given question, answer given in {language} language  in the following format :\
                Question:{question}\
                Answer:<answer> \
                Reasoning:<reasoning> in the context of the image in {language} language."

        # Construct conversation prompt
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant following user's instructions."}],
            },
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
        ]

        # Apply chat template to get the input text
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        # Generate answer
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generation = generation[0][input_len:]

        # Decode the generated answer and remove special tokens
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded if decoded != "" else "No answer generated"

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


# Main function to process dataset
def evaluate(model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str, language: str) -> None:
    """
    Process the dataset and generate answers for each question.

    Args:
        model: The loaded language model.
        processor: The associated processor.
        dataset (list): The dataset to process.
        image_folder (str): Path to the image folder.
        save_path (str): Path to save the results.
        language (str): Language in which the answer should be generated.

    Returns
    -------
        None
    """
    results = []
    logger.info("Starting evaluation...")
    intermediate_results_path = save_path.replace(".json", "_intermediate.json")
    prev_path = ""

    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            # print(data)
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")

            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue

            q_id = f"Question({language})"
            a_id = f"Answer({language})"
            answer = process_sample(model, processor, img_path, data[q_id], language)
            results.append(
                {
                    "ID": data["ID"],
                    "Question": data[q_id],
                    "Predicted_Answer": answer,
                    "Ground_Truth": data[a_id],
                    "Attribute": data["Attribute"],
                }
            )

            # Save intermediate results every 50
            if i % 10 == 0:
                intermediate_results_path = save_path.replace(
                    ".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(intermediate_results_path, "w") as f:
                    json.dump(results, f, indent=4, default=str, ensure_ascii=False)
                if prev_path != "":
                    os.remove(prev_path)
                logger.info(f"Intermediate results saved to {intermediate_results_path} and deleted {prev_path}.")
                prev_path = intermediate_results_path

            pbar.update(1)

    # Save final results to JSON file
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
    parser.add_argument("--save_path", type=str, help="Output file to save results")
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to process")
    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Identify language
    language = args.dataset.split("_")[-1].split(".")[0]
    logger.info(f"Processing dataset in {language} language...")

    # Load model
    model, processor = load_model(args.model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    if args.num_samples > 0:
        dataset = dataset[args.num_samples :]

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, language)
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

# To run the script:
# python gemma3_12b_eval2.py \
# --dataset <path_to_dataset_json> \
# --image_folder <path_to_image_folder> \
# --device <cuda_or_cpu> \
# --save_path <path_to_save_results_json> \
# --model_source <local_or_hf> \
# --num_samples <number_of_samples_to_process>

# Expect JSON with keys "ID", "Question", "Answer", and "Attribute". File pattern:
# "Eval2_<language>.json".

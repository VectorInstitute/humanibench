"""HumanI Bench — task 4 (multilingual): aya vision 8b eval3."""

import base64
import json
import logging
import os
import re
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

# Set environment variables for caching
OFFLOAD_FOLDER = ""  # Path to offload folder
os.environ["HF_HOME"] = ""  # Path where you want to store the huggingface cache
os.environ["TRANSFORMERS_CACHE"] = ""  # Path where you want to store the transformers cache


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """
    Load the Aya-Vision model and its processor.

    Args:
        model_source (str): 'local' to load from a local directory; otherwise, load from Hugging Face.

    Returns
    -------
        tuple: (model, processor)
    """
    print(f"Loading Aya-Vision Vision model from {'local directory' if model_source == 'local' else 'Hugging Face'}...")
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    # Ensure cache directories are set
    os.environ["HF_HOME"] = ""
    os.environ["TRANSFORMERS_CACHE"] = ""

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        offload_folder="/scratch/ssd004/scratch/aravindn/offload",
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
        PIL.Image.Image or None: Resized image if successful, None otherwise.
    """
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def extract_answer_and_reason(text: str) -> tuple[str, str | None]:
    """
    Extract the answer and reasoning from the VLM response.

    First attempts to extract a JSON substring from the response.
    If that fails, uses regex to extract the answer and reasoning.

    Args:
        text (str): The VLM response text.

    Returns
    -------
        tuple: (answer, reasoning) if extraction is successful, otherwise (text, None).
    """
    try:
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

    # Fallback: use regex to extract answer and reasoning
    try:
        pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip()
            reasoning = match.group(2).strip()
            return answer, reasoning
        return text, None
    except Exception:
        return text, None


def process_sample(model: Any, processor: Any, img_path: str, question: str, device: Any, language: str) -> str:
    """
    Process an image-question pair and generate a response.

    Args:
        model: Loaded model.
        processor: Associated processor.
        img_path (str): Path to the image file.
        question (str): Question text.
        device: Torch device.
        language (str): Prompt language.

    Returns
    -------
        str: Generated response or an error message.
    """
    try:
        # Resize the image
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        # Format the prompt
        query_prompt = f"""Answer the question using one of the given choices based on the image.

        Question ({language}):
        {question}

        Provide the response only in the following JSON format:
        {{"Answer": "The correct letter and option",
        "Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

        Do not provide any other extra information.
        """
        # Convert the image to bytes and then encode as base64
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
                    {"type": "text", "text": query_prompt},
                ],
            }
        ]

        # Prepare inputs using the chat template
        inputs = processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        # Generate the response
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode the generated tokens (excluding the input tokens)
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
            # Dynamically choose keys based on language
            q_key = f"Question({language})"
            o_key = f"Options({language})"
            a_key = f"Answer({language})"
            r_key = f"Reasoning({language})"

            # Check if required keys exist and are non-empty
            if q_key not in data or a_key not in data:
                logger.warning(f"Missing keys for {language} in dataset entry {data['ID']}")
                pbar.update(1)
                continue
            if not data[q_key] or not data[o_key]:
                logger.warning(f"Empty question or choices for {language} in dataset entry {data['ID']}")
                results.append(
                    {
                        "ID": data["ID"],
                        "Question": data[q_key],
                        "Predicted_Answer": None,
                        "Predicted_Reasoning": None,
                        "Ground_Truth_Answer": data[a_key],
                        "Ground_Truth_Reasoning": data[r_key],
                        "Attribute": data["Attribute"],
                    }
                )
                pbar.update(1)
                continue

            # Combine question and options
            question_with_choices = data[q_key] + "\n" + data[o_key]

            # Process the sample
            response = process_sample(
                model,
                processor,
                os.path.join(image_folder, f"{data['ID']}.jpg"),
                question_with_choices,
                device,
                language,
            )
            answer, reason = extract_answer_and_reason(response)
            results.append(
                {
                    "ID": data["ID"],
                    "Question": question_with_choices,
                    "Predicted_Answer": answer,
                    "Predicted_Reasoning": reason,
                    "Ground_Truth_Answer": data[a_key],
                    "Ground_Truth_Reasoning": data[r_key],
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
    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine language from dataset filename (assumes language is after the last '_' and before '.json')
    language = args.dataset.split("_")[-1].split(".")[0]
    logger.info(f"Processing dataset in {language} language...")

    # Load model and processor
    model, processor = load_model(args.model_source)
    # Note: The model handles device placement automatically via device_map
    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, language)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")


# The dataset should be of the name format: Eval3_<language>.json

# To run the script, use the following command:
# python aya_vision_8b_eval3.py \
# --dataset <path_to_dataset> \
# --image_folder <path_to_image_folder> \
# --device <device> \
# --save_path <path_to_save_results> \
# --model_source <local_or_hf>

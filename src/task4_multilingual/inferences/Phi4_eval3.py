"""HumanI Bench — task 4 (multilingual): Phi4 eval3."""

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
from transformers import AutoModelForCausalLM, AutoProcessor


# Model directory and Hugging Face model ID
MODEL_DIR = "/model-weights/Phi-4-multimodal-instruct"  # Local model path
HF_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"  # Hugging Face Model ID


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for Hugging Face cache directories
OFFLOAD_FOLDER = ""  # Path to offload folder
os.environ["HF_HOME"] = ""  # Path where you want to store the huggingface cache
os.environ["TRANSFORMERS_CACHE"] = ""  # Path where you want to store the transformers cache


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """
    Load the Phi-4 Vision model and its processor.

    Args:
        model_source (str): 'local' to load from a local directory, otherwise load from Hugging Face.

    Returns
    -------
        model: The loaded causal language model.
        processor: The associated processor.
    """
    source = "local directory" if model_source == "local" else "Hugging Face"
    print(f"Loading Phi-4 Vision model from {source}...")

    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Required for Phi-4
        torch_dtype="auto",  # Automatically selects best precision (FP16/BF16)
        device_map="auto",  # Automatically assigns to GPU
        _attn_implementation="eager",  # Default implementation
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """
    Resize an image to fit within max_size while preserving aspect ratio.

    Args:
        img_path (str): Path to the image file.
        max_size (tuple): Maximum (width, height).

    Returns
    -------
        Image: Resized PIL image if successful, None otherwise.
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

    The function first attempts to parse a JSON substring from the response.
    If that fails, it uses a regex pattern to extract the answer and reasoning.

    Args:
        text (str): The VLM response text.

    Returns
    -------
        tuple: (answer, reasoning) if extraction is successful, otherwise (text, None).
    """
    try:
        # Attempt to extract a JSON substring from the text.
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

    # Fallback: extract using regex pattern.
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


def process_sample(model: Any, processor: Any, img_path: str, question: str, language: str, device: Any) -> str:
    """
    Process a single image-question pair and generate a response.

    Args:
        model: The loaded language model.
        processor: The corresponding processor.
        img_path (str): Path to the image file.
        question (str): The question to be answered (with choices).
        language (str): Language for the prompt.
        device: Torch device to run the model on.

    Returns
    -------
        str: The generated response or an error message.
    """
    try:
        # Resize the image (do not save the resized copy)
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        user_prompt = f"""Answer the question using one of the given choices based on the image in the {language} language:
        Question ({language}):
        {question}

        Provide the response only in the following JSON format in {language} language:
        {{"Answer": "The correct letter and option in {language} language",
        "Reasoning": "A brief explanation (max 80 words) based on the details in the image in {language} language"}}.

        Do not provide any other extra information.
        """

        # Construct prompt with special tokens expected by the model
        prompt = f"<|user|><|image_1|>\n{user_prompt}<|end|><|assistant|>"

        # Prepare inputs and move them to the specified device
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

        # Remove input tokens from the output and decode the response
        generated_ids = output[:, inputs["input_ids"].shape[1] :]
        predicted_answer = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return predicted_answer if predicted_answer else "No answer generated"
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str, language: str) -> None:
    """
    Evaluate the model on a dataset of image-question pairs and save the results.

    Args:
        model: The loaded language model.
        processor: The corresponding processor.
        dataset (list): List of data samples.
        image_folder (str): Directory containing images.
        save_path (str): File path to save the final results.
        language (str): Language of the questions and answers.
        mode (str): Processing mode ('single' or 'batch').

    Returns
    -------
        None
    """
    results = []
    logger.info("Starting evaluation...")
    prev_path = ""

    # Process each sample with a progress bar.
    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                pbar.update(1)
                continue

            # Define keys dynamically based on language.
            q_key = f"Question({language})"
            o_key = f"Options({language})"
            a_key = f"Answer({language})"
            r_key = f"Reasoning({language})"

            if q_key not in data or a_key not in data:
                logger.warning(f"Question or Answer key not found in data: {data}")
                pbar.update(1)
                continue

            if not data[q_key] or not data[a_key]:
                logger.warning(f"Empty question or answer in data: {data}")
                results.append(
                    {
                        "ID": data["ID"],
                        "Question": data[q_key],
                        "Predicted_Answer": None,
                        "Predicted_Reasoning": None,
                        "Ground_Truth_Answer": data[a_key],
                        "Ground_Truth_Reasoning": data.get(r_key, None),
                        "Attribute": data["Attribute"],
                    }
                )
                pbar.update(1)
                continue

            question_with_choices = data[q_key] + "\n" + data[o_key]
            response = process_sample(
                model,
                processor,
                os.path.join(image_folder, f"{data['ID']}.jpg"),
                question_with_choices,
                language,
                device,
            )
            answer, reasoning = extract_answer_and_reason(response)
            results.append(
                {
                    "ID": data["ID"],
                    "Question": data[q_key],
                    "Prediction": response,
                    "Predicted_Answer": answer,
                    "Predicted_Reasoning": reasoning,
                    "Ground_Truth_Answer": data[a_key],
                    "Ground_Truth_Reasoning": data.get(r_key, None),
                    "Attribute": data["Attribute"],
                }
            )

            # Save intermediate results every 10 samples.
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

    # Save final results.
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Results saved to {save_path}.")


if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments.
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--image_folder", type=str, help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--save_path", type=str, help="Output file to save results")
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to process")
    args = parser.parse_args()

    # Define device globally.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Determine language from dataset filename (assumes language is after the last '_' and before '.json').
    language = args.dataset.split("_")[-1].split(".")[0]
    logger.info(f"Processing dataset in {language} language...")

    # Load model and processor.
    model, processor = load_model(args.model_source)
    model.to(device)

    # Load dataset.
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    if args.num_samples > 0:
        dataset = dataset[args.num_samples :]

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation.
    evaluate(model, processor, dataset, args.image_folder, args.save_path, language)
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

# To run the script:
# python Phi4_eval3.py \
#     --dataset <path_to_dataset_json> \
#     --image_folder <path_to_image_folder> \
#     --device <cuda_or_cpu> \
#     --save_path <output_file_path> \
#     --model_source <local_or_hf> \
#     --num_samples <number_of_samples_to_process>

# The dataset should include in-language questions, answers, and options. Name files
# "Eval3_<language>.json". Image filenames should match dataset IDs in image_folder.

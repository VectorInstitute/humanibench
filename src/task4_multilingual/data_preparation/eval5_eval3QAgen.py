"""HumanI Bench — task 4 (multilingual): eval5 eval3QAgen."""

import base64
import json
import os
from argparse import ArgumentParser
from typing import Any

from openai import OpenAI
from tqdm import tqdm


# OpenAI API Setup
def load_openai_client(api_key: str) -> Any:
    """
    Create and return an OpenAI client using the provided API key.

    Args:
        api_key (str): Your OpenAI API key.

    Returns
    -------
        OpenAI: An instance of the OpenAI client.
    """
    return OpenAI(api_key=api_key)


# Set batch size (max OpenAI batch size is 1000 per file)
BATCH_SIZE = 200  # Adjust as needed


def encode_image(image_path: str) -> str:
    """
    Encode an image file to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns
    -------
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_prompt(prompt_filepath: str) -> str:
    """
    Load and return a prompt from a text file.

    Args:
        prompt_filepath (str): Path to the prompt text file.

    Returns
    -------
        str: The loaded prompt, or a default prompt if file not found.
    """
    try:
        with open(prompt_filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_filepath}' not found.")
        return "Describe the given image."


def generate_batch_json_files(data: Any, language: str, folder: str) -> list[Any]:
    """
    Generate batch JSONL files for OpenAI's Batch API from the input data.

    Each item in the input data is expected to be a dictionary with keys:
    "ID", "Question", "Options", "Answer", "Reasoning", and "Attribute". For each
    item, this function constructs a prompt that instructs the model to generate a
    translated version in the target language.

    Args:
        data (list): List of dictionaries containing image QA data.
        language (str): Target language for the generated questions.

    Returns
    -------
        list: List of file paths to the generated batch JSONL files.
    """
    batch = []
    failed = []

    for item in tqdm(data, desc="Preparing Batch Requests"):
        try:
            entry_id = item["ID"]
            question = item["Question"]
            options = item["Options"]
            answer = item["Answer"]
            reasoning = item["Reasoning"]
            attribute = item["Attribute"]

            # folder = "/projects/NMB-Plus/E-VQA/data/processed_images"
            # image_path = os.path.join(folder, f"{entry_id}.jpg")
            # base64_image = encode_image(image_path)

            prompt = f"""**Prompt**:
            You are given a question, options, answer, and reasoning in English. Generate the corresponding question, options, answer, and reasoning in {language}. Ensure that your response is relevant, coherent, and maintains the context. Do not add any extra information or alter the meaning.

            **Example:**

            Input:
            Question (in English): “{question}”
            Options (in English): “{options}”
            Answer (in English): “{answer}”
            Reasoning (in English): “{reasoning}”

            Output:
            Q:<fill in the translated question>
            O:<fill in the translated options>
            A:<fill in the translated answer>
            R:<fill in the translated reasoning>

            Please stick to the format and ensure that the generated content in {language} are **exact translations** of the provided English text. Only generate the output part."""

            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            bat = {
                "custom_id": f"{entry_id}_{attribute}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o", "messages": prompt_messages, "max_tokens": 256},
            }
            batch.append(bat)
        except Exception as e:
            print(f"Error processing item {item['ID']}: {e}")
            failed.append(item["ID"])

    # Create output directory for batch files
    output_dir = f"batch_files_{language}"
    os.makedirs(output_dir, exist_ok=True)

    # Split into sub-batches of size BATCH_SIZE
    sub_batches = [batch[i : i + BATCH_SIZE] for i in range(0, len(batch), BATCH_SIZE)]
    batch_files = []
    for idx, sub_batch in enumerate(sub_batches):
        batch_filename = os.path.join(output_dir, f"batch_{idx}.jsonl")
        with open(batch_filename, "w") as f:
            for entry in sub_batch:
                f.write(json.dumps(entry) + "\n")
        batch_files.append(batch_filename)
        print(f"Saved batch {idx} with {len(sub_batch)} requests to {batch_filename}")

    # Save failed items
    with open(os.path.join(output_dir, "failed_requests.json"), "w") as f:
        json.dump(failed, f, indent=2)
    print(f"Generated {len(batch_files)} batch JSONL files.")

    return batch_files


def upload_batch_file(client: Any, batch_file: str) -> Any:
    """
    Upload a batch file to OpenAI for batch processing.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        batch_file (str): Path to the batch file.

    Returns
    -------
        str: The file ID of the uploaded batch file.
    """
    print(f"Uploading batch file: {batch_file} to OpenAI...")
    with open(batch_file, "rb") as file:
        response = client.files.create(file=file, purpose="batch")
    return response.id


def submit_batches(client: Any, batch_files: list[Any], language: str) -> None:
    """
    Submit batch jobs to OpenAI's Batch API using the uploaded batch files.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        batch_files (list): List of paths to batch files.
        language (str): Target language (used for naming the batch ID file).
    """
    batch_ids = []
    for batch_file in batch_files:
        try:
            file_id = upload_batch_file(client, batch_file)
            response = client.batches.create(
                input_file_id=file_id, endpoint="/v1/chat/completions", completion_window="24h"
            )
            batch_ids.append(response.id)
            print(f"Submitted {batch_file}, OpenAI Batch ID: {response.id}")
        except Exception as e:
            print(f"Failed to submit {batch_file}: {e}")

    # Save batch IDs to a file
    batch_id_file = f"batch_ids_{language}.txt"
    with open(batch_id_file, "w") as f:
        f.write("\n".join(batch_ids))
    print(f"Batch IDs saved to {batch_id_file}")


def parse_args() -> Any:
    """
    Parse command-line arguments.

    Returns
    -------
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser(
        description="Generate translated questions for images using OpenAI GPT-4 with batch processing."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to input JSON file with image QA data")
    parser.add_argument("--language", type=str, default="Urdu", help="Target language for translation")
    parser.add_argument("--openai_api_key", type=str, required=True, help="Your OpenAI API key")
    parser.add_argument("--images_folder", type=str, default="images", help="Folder containing image files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    client = load_openai_client(args.openai_api_key)

    # Load input data (a list of dictionaries)
    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Generate batch JSONL files
    batch_files = generate_batch_json_files(data, args.language, folder=args.images_folder)

    # Submit batches to OpenAI's Batch API
    submit_batches(client, batch_files, language=args.language)

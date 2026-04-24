"""HumanI Bench — task 4 (multilingual): eval5 eval2QAgen."""

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
    Load and return an OpenAI client using the provided API key.

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
        str: The loaded prompt.
    """
    try:
        with open(prompt_filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_filepath}' not found.")
        return "Describe the given image."


def generate_batch_json_files(data: Any, language: str, folder: str) -> list[Any]:
    """
    Generate batch JSONL files for OpenAI's Batch API from input data.

    Each item in the input data should be a dictionary containing keys:
    "ID", "Question", "Answer", and "Attribute". The function encodes the
    corresponding image (using its ID) to base64 and constructs a request payload.

    Args:
        data (list): List of dictionaries with image QA data.
        language (str): Target language for the generated questions.
        folder (str): Path to the folder containing the images.

    Returns
    -------
        list: List of file paths to the generated batch JSONL files.
    """
    batch = []
    failed = []

    # Process each item in the data
    for item in tqdm(data, desc="Preparing Batch Requests"):
        try:
            entry_id = item["ID"]
            question = item["Question"]
            answer = item["Answer"]
            attribute = item["Attribute"]
            # image_path = os.path.join(folder, f"{entry_id}.jpg")
            # base64_image = encode_image(image_path)

            prompt = f"""**Prompt**:
            You are given a question and answer in English. Based on the content of the image and the provided English question and answer, generate the corresponding question and answer in {language}. Ensure that your response is relevant, coherent, and maintains the context. Do not add any extra information or alter the meaning of the question or answer.

            **Example:**

            Input:
            Question (in English): “{question}”
            Answer (in English): “{answer}”

            Output:
            Question (in {language}): <fill in the translated question>
            Answer (in {language}): <fill in the translated answer>

            Please stick to the format and ensure that the generated question and answer in {language} are **exact translations** of the provided English question and answer. Only generate the output part, not the input part. Do not add extra details or change the context of the question or answer."""

            # Construct the prompt messages for the batch request
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Construct OpenAI Batch API request entry
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
    os.makedirs(f"batch_files_{language}", exist_ok=True)

    # Split batch into sub-batches (each of maximum size BATCH_SIZE)
    sub_batches = [batch[i : i + BATCH_SIZE] for i in range(0, len(batch), BATCH_SIZE)]
    batch_files = []
    for idx, sub_batch in enumerate(sub_batches):
        batch_filename = f"batch_files_{language}/batch_{idx}.jsonl"
        with open(batch_filename, "w") as f:
            for entry in sub_batch:
                f.write(json.dumps(entry) + "\n")
        batch_files.append(batch_filename)
        print(f"Saved batch {idx} with {len(sub_batch)} requests to {batch_filename}")

    # Save failed item IDs
    with open(f"batch_files_{language}/failed_requests.json", "w") as f:
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


def submit_batches(client: Any, batch_files: list[Any], batch_id_file: str = "batch_ids.txt") -> None:
    """
    Submit batch jobs to OpenAI's Batch API using the uploaded batch files.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        batch_files (list): List of batch file paths.
        batch_id_file (str): File to save the resulting batch IDs.
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

    # Save batch IDs to file
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
    parser = ArgumentParser(description="Generate questions for images using OpenAI GPT-4 with batch processing.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON file with input data")
    parser.add_argument("--language", type=str, default="Portuguese", help="Language for the generated questions")
    parser.add_argument("--openai_api_key", type=str, required=True, help="Your OpenAI API key")
    parser.add_argument(
        "--image_folder", type=str, default="./data/processed_images", help="Path to the folder containing images"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    client = load_openai_client(args.openai_api_key)

    # Load input data (list of dictionaries)
    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Generate batch JSONL files for processing
    batch_files = generate_batch_json_files(data, args.language, args.image_folder)

    # Submit batches to OpenAI's Batch API
    submit_batches(client, batch_files)

# To run the script:
# python eval5_eval2QAgen.py \
#     --data_path <path_to_your_json_file> \
#     --language <language> \
#     --openai_api_key <your_openai_api_key> \
#     --image_folder <path_to_your_image_folder>

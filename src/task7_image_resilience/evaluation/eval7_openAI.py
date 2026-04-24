"""HumanI Bench — task 7: OpenAI batch evaluation for image resilience."""

import base64
import json
import os
from argparse import ArgumentParser
from typing import Any

from openai import OpenAI
from tqdm import tqdm


# OpenAI API Setup
def load_openai_client(api_key: str) -> Any:
    """Load openai client."""
    return OpenAI(api_key=api_key)


# Set batch size (max OpenAI batch size is 1000 per file)
BATCH_SIZE = 250  # Adjust as needed


def encode_image(image_path: str) -> str:
    """
    Encode an image as a base64 string.

    Args:
    - image_path (str): The path to the image to be encoded.

    Returns
    -------
    - str: The base64 encoded image string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Load the prompt from prompt.txt
def load_prompt(prompt_filepath: str) -> str:
    """
    Load prompt text from a file.

    Args:
    - prompt_filepath (str): The file path for the prompt text file.

    Returns
    -------
    - str: The loaded prompt text.
    """
    try:
        with open(prompt_filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_filepath}' not found.")
        return "Describe the given image."


# Generate batch JSONL files for OpenAI's Batch API
def generate_batch_json_files(data: list[dict[str, Any]]) -> list[str]:
    """
    Build JSONL batch files for OpenAI API requests.

    Args:
    - data (list): List of dictionaries containing the data to be processed.

    Returns
    -------
    - list: List of generated batch files.
    """
    batch = []
    failed = []

    for item in tqdm(data, desc="Preparing Batch Requests"):
        try:
            id = item["ID"]
            question = item["Question"]
            model_answer = item["Model_Answer"]
            ground_truth = item["Ground_Truth"]
            attribute = item["Attribute"]
            image_id = item["ID"]
            folder = "/projects/NMB-Plus/E-VQA/data/eval7/processed_images/subsample"
            image_path = os.path.join(folder, f"{image_id}.jpg")
            base64_image = encode_image(image_path)

            prompt = f"""
                    You are an evaluation agent. Your task is to compare a potentially degraded answer due to perturbated images such as noise, blurriness, or other distortions, with a reference answer generated from a clean image answering a question. You will be provided with a question, a degraded answer generated from a potentially degraded image, and a reference answer generated from a clean image. Your task is to evaluate the match between the degraded answer and the reference answer.

                    Focus on identifying whether any important **visual details** (such as logos, text, icons, or other distinguishing elements) are present in the reference image but **missing or distorted** in the degraded image, and how this might affect the generated answer. Consider whether these missing or altered details lead to a significant change in the interpretation of the image, and whether the model's answer reflects this absence.

                    {{
                        "question": "{question}",
                        "reference_answer": "{ground_truth}",
                        "degraded_answer": "{model_answer}"
                    }}


                    Provide the output in the following JSON format:
                    ```json
                    {{
                    "match": "Yes" or "No",
                    "score": <integer from 1 to 10 based how much the degraded answer matches the reference answer>,
                    "missing_details": <list of important visual details that are missing or distorted in the degraded answer if any>,
                    "explanation": "<brief explanation of the match or mismatch, and if any key details (e.g., logos, text) are missing or distorted in the degraded answer, written in 50 words or less>"
                    }}
                    """

            # Construct the prompt
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ]

            # Construct OpenAI Batch API request format
            bat = {
                "custom_id": f"{id}_{attribute}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o-mini", "messages": prompt_messages, "max_tokens": 256},
            }
            batch.append(bat)

        except Exception as e:
            print(f"Error processing item {item['ID']}: {e}")
            failed.append(item["ID"])

    # Create output directory
    os.makedirs("batch_files", exist_ok=True)

    # Split into sub-batches (max OpenAI batch size is 1000 per file)
    sub_batches = [batch[i : i + BATCH_SIZE] for i in range(0, len(batch), BATCH_SIZE)]

    # Save batch JSONL files
    batch_files = []
    for idx, sub_batch in enumerate(sub_batches):
        batch_filename = f"batch_files/batch_{idx}.jsonl"
        with open(batch_filename, "w") as f:
            for entry in sub_batch:
                f.write(json.dumps(entry) + "\n")

        batch_files.append(batch_filename)
        print(f"Saved batch {idx} with {len(sub_batch)} requests to {batch_filename}")

    # Save failed items
    with open("batch_files/failed_requests.json", "w") as f:
        json.dump(failed, f, indent=2)

    print(f"Generated {len(batch_files)} batch JSONL files.")

    return batch_files


# Upload batch file to OpenAI
def upload_batch_file(client: Any, batch_file: str) -> Any:
    """
    Upload a batch file to OpenAI.

    Args:
    - client: The OpenAI client instance.
    - batch_file (str): The batch file path.

    Returns
    -------
    - str: The ID of the uploaded batch file.
    """
    print(f"Uploading batch file: {batch_file} to OpenAI...")
    with open(batch_file, "rb") as file:
        response = client.files.create(file=file, purpose="batch")
    return response.id  # Returns the uploaded file's ID


# Submit batch jobs to OpenAI's Batch API
def submit_batches(client: Any, batch_files: list[Any], batch_id_file: str = "batch_ids.txt") -> None:
    """
    Submit batch files to the OpenAI Batch API.

    Args:
    - client: The OpenAI client instance.
    - batch_files (list): List of batch files to submit.
    - batch_id_file (str): The file to save the batch IDs.
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
    with open(batch_id_file, "w") as f:
        f.write("\n".join(batch_ids))


# Argument parsing
def parse_args() -> Any:
    """Parse args."""
    parser = ArgumentParser(description="Generate questions for images using OpenAI GPT-4 with batch processing.")
    parser.add_argument("--data_folder", type=str, help="Path to the folder containing images and JSON files.")
    parser.add_argument("--openai_api_key", type=str, required=True, help="Your OpenAI API key")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    client = load_openai_client(args.openai_api_key)

    data_path = args.data_folder
    # Get all json files in the folder
    json_files = [f for f in os.listdir(data_path) if f.endswith(".json")]
    for json_file in json_files:
        print(json_file)
        # continue

        with open(os.path.join(data_path, json_file), "r") as f:
            data = json.load(f)
        # Print number of items in the data
        print(f"Number of items in {json_file}: {len(data)}")

        batch_files = generate_batch_json_files(data)
        print("Generated batch files.")

        input_file_name = os.path.basename(data_path).replace(".json", "")
        batch_id_file = f"batch_ids_{input_file_name}.txt"
        # Submit batches to OpenAI
        submit_batches(client, batch_files, batch_id_file)
        print("Submitted batches to OpenAI.")

# To run the script:
# python eval7_openAI.py --data_folder <path_to_your_data_folder> --openai_api_key <your_openai_api_key>

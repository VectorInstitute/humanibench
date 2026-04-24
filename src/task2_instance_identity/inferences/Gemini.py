"""HumanI Bench — task 2 (instance identity): Gemini."""

import csv
import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

from google import genai
from google.genai import types
from tqdm import tqdm


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Gemini API key from the environment variable
MODEL_NAME = "gemini-2.0-flash-001"  # Change to "gemini-2.5-pro-preview-03-25" as needed
# Only run this block for Gemini Developer API
client = genai.Client(
    vertexai=True,
    project="",  # Change to your project name
    location="",  # Change to your project location
)


def encode_image(image_path: str) -> bytes:
    """Encode image."""
    with open(image_path, "rb") as image_file:
        return image_file.read()


def extract_answer_and_reasoning(response: str) -> tuple[str | None, str | None]:
    # Check if the response contains both 'Answer' and 'Reasoning' sections
    """Extract answer and reasoning."""
    try:
        if "Answer:" in response and "Reasoning:" in response:
            answer, reasoning = response.split("Reasoning:", 1)
            return answer.strip(), reasoning.strip()
        logging.error("Response format is incorrect. Expected 'Answer: <answer> Reasoning: <reasoning>'")
        logging.error(f"Received response: {response}")
        return response.strip(), "No reasoning provided"
    except Exception as e:
        logging.error(f"Error processing response: {e}")
        return None, None


def process_data(data: list[Any], image_folder: str, output_path: str = "gemini_results.csv") -> list[Any]:
    """Process data."""
    results = []
    failed = []

    # Intermediate path is same as output_path_intermediate
    intermediate_path = os.path.splitext(output_path)[0] + "_intermediate.json"
    if os.path.exists(intermediate_path):
        with open(intermediate_path, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} previously processed items.")
        processed_ids = {(item["ID"], item["Attribute"]) for item in results}
        print(f"Loaded {len(processed_ids)} processed IDs.")
    else:
        results = []
        processed_ids = set()

    for item in tqdm(data, desc="Sending to Gemini 2.5 Pro"):
        try:
            id = item["ID"]
            question = item["Question"]
            attribute = item["Attribute"]
            image_id = item["ID"]
            answer = item["Answer"]
            image_path = os.path.join(image_folder, f"{image_id}.jpg")
            image_bytes = encode_image(image_path)

            # Check if the tuple of id, attribute is already processed
            if (id, attribute) in processed_ids:
                print(f"Skipping already processed item {id} with attribute {attribute}")
                continue
            prompt = (
                f"Given question, answer in the following format:\n"
                f"Question: {question}\n"
                f"Answer: <answer>\n"
                f"Reasoning: <reasoning> in the context of the image."
            )

            # Create a Part from the image bytes (no need to base64-encode)
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            )

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        image_part,
                    ],
                )
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=0.0,
                # top_p = 0.95,
                max_output_tokens=256,
                response_modalities=["TEXT"],
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            )

            full_response = ""
            for chunk in client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    # print(f"Chunk: {chunk.text}")
                    full_response += chunk.text  # accumulate the response
            generated_answer = full_response

            # Remove the question from the generated answer
            generated_answer = generated_answer.replace(question, "")
            # Remove Question: \n
            generated_answer = generated_answer.replace("Question: \n", "")
            pred_answer, pred_reasoning = extract_answer_and_reasoning(generated_answer)
            # Remove Answer: from pred_answer
            pred_answer = pred_answer.replace("Answer:", "").strip() if pred_answer is not None else ""

            # Append the result
            results.append(
                {
                    "ID": id,
                    "Attribute": attribute,
                    "Predicted_Answer": generated_answer,
                    "Question": question,
                    "Answer": pred_answer,
                    "Reasoning": pred_reasoning,
                    "Ground_Truth": answer,
                }
            )

            # Save intermediate results
            with open(intermediate_path, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"Error processing item {item['ID']}: {e}")
            failed.append(item)
            continue

    # Save results to CSV
    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["ID", "Attribute", "Predicted_Answer", "Question", "Answer", "Reasoning", "Ground_Truth"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Save in JSON format
    # Get the path removing rightmost part of the filename
    path = os.path.dirname(output_path)
    # Get the filename without extension
    filename = os.path.basename(output_path).split(".")[0]
    # Save the results in JSON format
    json_path = os.path.join(path, f"{filename}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {json_path}")

    # Save failed
    with open("failed_requests.json", "w") as f:
        json.dump(failed, f, indent=4)

    print(f"Saved {len(results)} results to {output_path}")
    return results


def main() -> None:
    """Run the main workflow."""
    parser = ArgumentParser(description="Process images and questions with Gemini 2.5 Pro.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--image_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file.")
    args = parser.parse_args()

    # Load the dataset
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # Select first sample for testing
    # data = data[:1]

    # Process the data
    process_data(data, args.image_folder, args.output_file)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

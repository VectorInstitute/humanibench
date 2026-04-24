"""HumanI Bench — task 5 (visual grounding): gemini detection."""

import csv
import json
import os

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.0-flash-001"


def run_inference(
    processed_file: str,
    images_dir: str,
    results_file: str,
    cache_dir: str | None = None,
) -> None:
    """Run inference."""
    with open(processed_file, "r") as f:
        data = json.load(f)

    client = genai.Client(
        vertexai=True,
        project=os.getenv("GEMINI_PROJECT", ""),
        location=os.getenv("GEMINI_LOCATION", ""),
    )

    def encode_image(image_path: str) -> bytes:
        with open(image_path, "rb") as f_img:
            return f_img.read()

    def extract_answer_and_reasoning(response: str) -> tuple[str | None, str | None]:
        try:
            if "Answer:" in response and "Reasoning:" in response:
                answer, reasoning = response.split("Reasoning:", 1)
                return answer.strip().replace("Answer:", "").strip(), reasoning.strip()
            return response.strip(), "No reasoning provided"
        except Exception:
            return None, None

    results = []
    failed = []
    intermediate_path = os.path.splitext(results_file)[0] + "_intermediate.json"

    processed_ids = set()
    if os.path.exists(intermediate_path):
        with open(intermediate_path, "r") as f:
            results = json.load(f)
        processed_ids = {(item["ID"], item["Attribute"]) for item in results}

    for item in data:
        try:
            id = item["ID"]
            attribute = item["Attribute"]
            question = item["Question"]
            answer = item["Answer"]

            if (id, attribute) in processed_ids:
                print(f"Skipping already processed: {id} | {attribute}")
                continue

            image_path = os.path.join(images_dir, f"{id}.jpg")
            image_bytes = encode_image(image_path)

            prompt = (
                f"Given question, answer in the following format:\n"
                f"Question: {question}\n"
                f"Answer: <answer>\n"
                f"Reasoning: <reasoning> in the context of the image."
            )

            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(prompt),
                        image_part,
                    ],
                )
            ]

            generate_config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                response_modalities=["TEXT"],
                safety_settings=[
                    types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
                    for c in [
                        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    ]
                ],
            )

            full_response = ""
            for chunk in client.models.generate_content_stream(
                model=MODEL_NAME, contents=contents, config=generate_config
            ):
                if chunk.text:
                    full_response += chunk.text

            full_response = full_response.replace(question, "").replace("Question: \n", "")
            pred_answer, pred_reasoning = extract_answer_and_reasoning(full_response)

            results.append(
                {
                    "ID": id,
                    "Attribute": attribute,
                    "Predicted_Answer": full_response,
                    "Question": question,
                    "Answer": pred_answer,
                    "Reasoning": pred_reasoning,
                    "Ground_Truth": answer,
                }
            )

            with open(intermediate_path, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"Error processing {item['ID']}: {e}")
            failed.append(item)
            continue

    # Save results
    with open(results_file, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["ID", "Attribute", "Predicted_Answer", "Question", "Answer", "Reasoning", "Ground_Truth"],
            quotechar='"',
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = os.path.splitext(results_file)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    with open("failed_requests.json", "w") as f:
        json.dump(failed, f, indent=4)

    print(f"✅ Saved {len(results)} results to {results_file}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Run inference using Gemini 2 Vision API.")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the output CSV.")
    parser.add_argument("--cache_dir", type=str, default="", help="(Unused for Gemini, kept for compatibility.)")

    args = parser.parse_args()
    start = time.time()
    run_inference(
        processed_file=args.processed_file,
        images_dir=args.images_dir,
        results_file=args.results_file,
        cache_dir=args.cache_dir,
    )
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

"""Convert Eval3 results to the desired format."""

import argparse
import json
import os
import re

from langdetect import LangDetectException, detect


# Utility function to check if text is majority English
def is_majority_english(text: str, threshold: float = 0.5) -> bool:
    """Is majority english."""
    try:
        detected_language = detect(text)
        return detected_language == "en"
    except LangDetectException:
        words_in_text = re.findall(r"\b\w+\b", text)
        if not words_in_text:
            return False
        english_like_words = re.findall(r"\b[a-zA-Z]+\b", text)
        return (len(english_like_words) / len(words_in_text)) > threshold


# Attempt to fix truncated JSON (missing quotes, commas, or closing braces)
def fix_truncated_json(predicted_answer: str) -> str:
    """Fix truncated json."""
    if predicted_answer.strip().startswith("{"):
        if not predicted_answer.strip().endswith("}"):
            predicted_answer = predicted_answer.strip() + "}"

        reasoning_start_index = predicted_answer.find('"Reasoning":') + len('"Reasoning":')
        reasoning_part = predicted_answer[reasoning_start_index:].strip()

        if reasoning_part and not reasoning_part.endswith('"'):
            predicted_answer = predicted_answer.rstrip() + '"'

        try:
            parsed_answer = json.loads(predicted_answer)
            return json.dumps(parsed_answer)
        except json.JSONDecodeError:
            closing_brace_index = predicted_answer.rfind("}")
            return predicted_answer[: closing_brace_index + 1]

    return predicted_answer


# Extract answer and reasoning from the predicted answer
def extract_answer_and_reasoning(predicted_answer: str) -> tuple[str | None, str | None]:
    """Extract answer and reasoning."""
    if predicted_answer is None:
        return "", None

    try:
        if isinstance(predicted_answer, str) and predicted_answer.strip().startswith("{"):
            predicted_answer = fix_truncated_json(predicted_answer)
            try:
                parsed_answer = json.loads(predicted_answer)
                answer = parsed_answer.get("Answer", "").strip() if isinstance(parsed_answer, dict) else ""
                reasoning = parsed_answer.get("Reasoning", "").strip() if isinstance(parsed_answer, dict) else ""
            except json.JSONDecodeError:
                answer = ""
                reasoning = None
        else:
            answer = predicted_answer.strip()
            reasoning = None
    except json.JSONDecodeError:
        answer = predicted_answer.strip()
        reasoning = None

    return answer, reasoning


# Convert the input file into the desired format
def convert_to_lang_eval3(input_file_path: str, output_folder_path: str) -> str:
    """Convert to lang eval3."""
    with open(input_file_path, "r") as f:
        data = json.load(f)

    file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_folder_path, file_name)

    # Prepare output data
    data_output = []
    for item in data:
        predicted_answer = item["Predicted_Answer"]
        answer, reasoning = extract_answer_and_reasoning(predicted_answer)

        if answer is not None and is_majority_english(answer):
            answer = ""
        if reasoning and is_majority_english(reasoning):
            reasoning = None

        data_output.append(
            {
                "ID": item["ID"],
                "Question": item["Question"],
                "Predicted_Answer": answer,
                "Predicted_Reasoning": reasoning,
                "Ground_Truth_Answer": item["Ground_Truth_Answer"],
                "Ground_Truth_Reasoning": item["Ground_Truth_Reasoning"],
                "Attribute": item["Attribute"],
            }
        )

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data_output, f, indent=4, ensure_ascii=False)

    print(f"Saved to {output_file_path}")
    return output_file_path


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a JSON file to a JSON file with utf-8 encoding")
    parser.add_argument(
        "--input_folder_path", type=str, default="./eval5/evaluation/results/Eval3", help="The input folder path"
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="./eval5/evaluation/results/Eval3_decoded",
        help="The output folder path",
    )
    args = parser.parse_args()

    # Get all files in input folder
    input_files = [f for f in os.listdir(args.input_folder_path) if f.endswith(".json")]

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    # Process each input file
    for input_file in input_files:
        input_file_path = os.path.join(args.input_folder_path, input_file)
        convert_to_lang_eval3(input_file_path, args.output_folder_path)
        print(f"Converted {input_file}")

    print("Done")

# To run the script, use the command:
# python convert_to_lang_eval3.py \
#     --input_folder_path <path_to_input_folder> \
#     --output_folder_path <path_to_output_folder>
# This script converts and cleans input JSON files, then saves them to the output folder.

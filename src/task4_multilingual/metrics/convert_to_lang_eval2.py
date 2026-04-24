"""Convert Eval2 results to the desired format."""

import argparse
import json
import os
import re
from typing import Any

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


# Process data using common logic
def process_data(data: Any, file_name: str, processor_function: Any) -> Any:
    """Process data."""
    results = []
    for entry in data:
        predicted_answer = entry["Predicted_Answer"]

        # Process reasoning and answer
        predicted_answer, reasoning = processor_function(predicted_answer)

        # Clean predicted_answer and reasoning
        predicted_answer = clean_text(predicted_answer)
        if reasoning:
            reasoning = clean_text(reasoning)

        # Language detection logic
        if is_majority_english(predicted_answer):
            predicted_answer = ""
        if reasoning and is_majority_english(reasoning):
            reasoning = None

        # Append the processed results
        results.append(
            {
                "ID": entry["ID"],
                "Question": entry["Question"],
                "Predicted_Answer": entry["Predicted_Answer"],
                "Model_Answer": predicted_answer,
                "Model_Reasoning": reasoning,
                "Ground_Truth": entry["Ground_Truth"],
                "Attribute": entry["Attribute"],
            }
        )

    return results


# Function to process specific parts of the data
def process_answer_and_reasoning(predicted_answer: str) -> tuple[str, str | None]:
    """Process answer and reasoning."""
    if "Reasoning:" in predicted_answer:
        predicted_answer, reasoning = predicted_answer.split("Reasoning:", 1)
        reasoning = reasoning.strip()
    else:
        reasoning = None

    if "Answer:" in predicted_answer:
        predicted_answer = predicted_answer.split("Answer:")[1].strip()

    return predicted_answer, reasoning


# Clean up unwanted characters from the text
def clean_text(text: str) -> str:
    # Remove unwanted characters
    """Clean text."""
    text = re.sub(r"[<>]", "", text)  # remove < > tags
    text = re.sub(r"\*\*", "", text)  # remove bold formatting
    text = re.sub(r"\n\n", "", text)  # remove unnecessary newlines
    text = re.sub(r"\n", "", text)  # remove single newlines
    return text.strip()


# Main function to convert input to the desired format
def convert_to_lang_eval2(input_file_path: str, output_folder_path: str) -> str | None:
    """Convert to lang eval2."""
    with open(input_file_path, "r") as f:
        data = json.load(f)

    file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_folder_path, file_name)

    # Determine the appropriate processing function based on the file name
    if "Aya_Vision" in file_name or "Llama3_2" in file_name or "Phi4" in file_name or "gemma3_12b" in file_name:
        data_output = process_data(data, file_name, process_answer_and_reasoning)
    else:
        print("Invalid file name")
        return None

    # Save the processed data to output folder
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data_output, f, indent=4, ensure_ascii=False)

    print(f"Saved to {output_file_path}")
    return output_file_path


# Main script to handle argument parsing and file processing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a json file to a json file with utf-8 encoding")
    parser.add_argument(
        "--input_folder_path", type=str, default="./eval5/evaluation/results/Eval2", help="The input folder path"
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="./eval5/evaluation/results/Eval2_decoded",
        help="The output folder path",
    )
    args = parser.parse_args()

    # Get all files in input folder
    input_files = os.listdir(args.input_folder_path)

    # Process each file
    for input_file in input_files:
        input_file_path = os.path.join(args.input_folder_path, input_file)
        convert_to_lang_eval2(input_file_path, args.output_folder_path)
        print(f"Converted {input_file}")

    print("Done")

# To run the script, use the command:
# python convert_to_lang_eval2.py \
#     --input_folder_path <path_to_input_folder> \
#     --output_folder_path <path_to_output_folder>
# Note: The script processes input JSON files, extracts info, cleans it, and saves to the output folder.

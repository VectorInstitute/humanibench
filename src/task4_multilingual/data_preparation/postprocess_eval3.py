"""HumanI Bench — task 4 (multilingual): postprocess eval3."""

import argparse
import glob
import json
import os
from typing import Any


def process_jsonl_files(jsonl_files: list[Any], output_json_file: str) -> None:
    """
    Process JSONL files and merge them into a single output JSON.

    For each JSONL file, the function reads each line, parses it as JSON, extracts the
    'custom_id' field (splitting it into 'ID' and 'Attribute') and the content from the
    response (if available). The combined results are then written to the output JSON file.

    Args:
        jsonl_files (list): List of paths to JSONL files.
        output_json_file (str): Path to the output JSON file.

    Returns
    -------
        None
    """
    results = []

    # Process each JSONL file in the list
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    entry = json.loads(line.strip())

                    # Extract custom_id and split into ID and Attribute
                    custom_id = entry.get("custom_id", "")
                    if custom_id:
                        id_part, attribute_part = custom_id.split("_", 1)
                    else:
                        id_part, attribute_part = "", ""

                    # Extract content from the response
                    content = ""
                    if entry.get("response") and entry["response"].get("body"):
                        body = entry["response"]["body"]
                        choices = body.get("choices", [{}])
                        content = choices[0].get("message", {}).get("content", "")

                    # Append the processed entry to the results list
                    results.append({"ID": id_part, "Attribute": attribute_part, "content": content})
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line: {line}")
                except Exception as e:
                    print(f"An error occurred while processing the file: {e}")

    # Write the combined results to the output JSON file (UTF-8 encoding)
    with open(output_json_file, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_json_file}")


def combine_json_data(output_dir: str, selected_file: str, base_file: str, output_file: str, language: str) -> None:
    """
    Combine language-specific JSON data with base English JSON data.

    This function reads a language-specific JSON file and a base English JSON file,
    matches entries based on 'ID' and 'Attribute', and creates a combined entry that
    includes both English and language-specific fields (Question, Options, Answer, Reasoning).
    The combined data is saved to an output file.

    Args:
        output_dir (str): Directory containing language JSON files (not used in this function).
        selected_file (str): Path to the language-specific JSON file.
        base_file (str): Path to the base English JSON file.
        output_file (str): Path to save the combined output JSON file.
        language (str): Language identifier (e.g., Bengali).
    """
    # Load the selected file (language-specific data)
    with open(selected_file, "r", encoding="utf-8") as f:
        selected_data = json.load(f)
    print(f"Number of selected entries (language-specific) from {selected_file}: {len(selected_data)}")

    # Load the base file (English data)
    with open(base_file, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    print(f"Number of entries in base file (English data) from {base_file}: {len(base_data)}")

    combined_data = []

    # Iterate over each entry in the base (English) file
    for english_entry in base_data:
        entry_id = english_entry["ID"]
        entry_attribute = english_entry["Attribute"]

        # Search for the matching entry in the selected (language-specific) file
        matching_entry = None
        for language_entry in selected_data:
            if language_entry["ID"] == entry_id and language_entry["Attribute"] == entry_attribute:
                matching_entry = language_entry
                break

        # If a matching entry is found, combine the data
        if matching_entry:
            combined_entry = {
                "ID": entry_id,
                "Attribute": entry_attribute,
                "Question(English)": english_entry["Question"],
                "Options(English)": english_entry["Options"],
                "Answer(English)": english_entry["Answer"],
                "Reasoning(English)": english_entry["Reasoning"],
                "Language": language,
                f"Question({language})": matching_entry["Question"],
                f"Answer({language})": matching_entry["Answer"],
                f"Options({language})": matching_entry["Options"],
                f"Reasoning({language})": matching_entry["Reasoning"],
            }
            combined_data.append(combined_entry)
        else:
            print(f"No match found for ID: {entry_id}, Attribute: {entry_attribute} in the selected file")

    # Save the combined data to the output JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"Combined data saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process JSONL files and combine with base English JSON data.")

    parser.add_argument("--jsonl_dir", required=True, help="Directory containing .jsonl files to process.")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed and combined JSON files.")
    parser.add_argument(
        "--input_folder", required=True, help="Directory containing language-specific processed JSON files."
    )
    parser.add_argument("--base_file", required=True, help="Path to the base English JSON file.")

    args = parser.parse_args()

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process JSONL files
    jsonl_files = glob.glob(os.path.join(args.jsonl_dir, "*.jsonl"))
    output_files = [
        os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(file))[0]}_processed.json")
        for file in jsonl_files
    ]

    for input_file, output_file in zip(jsonl_files, output_files):
        process_jsonl_files([input_file], output_file)

    # Combine processed language JSONs with base English data
    input_files = os.listdir(args.input_folder)

    for selected_file in input_files:
        language = selected_file.split("_")[1]  # Careful: assumes filename format like 'something_LANG.json'
        selected_file_path = os.path.join(args.input_folder, selected_file)
        combined_output_file = os.path.join(args.output_dir, f"Eval3_{language}_combined.json")
        combine_json_data(args.output_dir, selected_file_path, args.base_file, combined_output_file, language)


# To run the script,
# python postprocess_eval3.py \
#     --jsonl_dir "./your/jsonl_folder" \
#     --output_dir "./your/output_folder" \
#     --input_folder "./your/input_folder_with_translations" \
#     --base_file "./your/base/Selected_QA_Eval3.json"

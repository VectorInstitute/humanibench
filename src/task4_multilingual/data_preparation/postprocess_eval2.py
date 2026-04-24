"""HumanI Bench — task 4 (multilingual): postprocess eval2 JSON."""

import argparse
import glob
import json
import os
import re
from typing import Any


def process_jsonl_files(jsonl_files: list[Any], output_json_file: str) -> None:
    """
    Process a list of JSONL files and combine their entries into a single JSON file.

    Each line in a JSONL file should be a JSON object containing a "custom_id" field,
    which is split into "ID" and "Attribute", and a "response" field from which content
    is extracted.

    Args:
        jsonl_files (list): List of paths to JSONL files to be processed.
        output_json_file (str): Path to the output JSON file where results are saved.

    Returns
    -------
        None. The combined results are written to the output JSON file.
    """
    results = []

    # Iterate over each JSONL file in the provided list
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    entry = json.loads(line.strip())

                    # Extract custom_id and split it into ID and Attribute
                    custom_id = entry.get("custom_id", "")
                    if custom_id:
                        id_part, attribute_part = custom_id.split("_", 1)
                    else:
                        id_part, attribute_part = "", ""

                    # Extract content from the response if available
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

    # Write the combined results to the output JSON file
    with open(output_json_file, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_json_file}")


# Configuration for language-specific processing.
# For each language, we define:
# - remove_pattern: A regex pattern to remove from extracted question/answer.
# - alt_extraction: A tuple (question_marker, answer_marker) to use if the default extraction fails.
#   If answer_marker is None, only the question is extracted.
# - prefix_removals (optional): A list of prefixes to remove from the question.
LANGUAGE_CONFIG: dict[str, dict[str, Any]] = {
    "Urdu": {"remove_pattern": r"\(in Urdu\)", "alt_extraction": None},
    "Tamil": {"remove_pattern": r"\(in Tamil\)", "alt_extraction": None},
    "Spanish": {
        "remove_pattern": r"\(in Spanish\)",
        "alt_extraction": ("Pregunta (en español):", "Respuesta (en español):"),
    },
    "Punjabi": {
        "remove_pattern": r"\(in Punjabi\)",
        "alt_extraction": ("Pregunta (en español):", "Respuesta (en español):"),
    },
    "Portuguese": {
        "remove_pattern": r"\(in Portuguese\)",
        "alt_extraction": ("Questão (em Português):", "Resposta (em Português):"),
    },
    "Persian": {"remove_pattern": r"\(in Persian\)", "alt_extraction": ("پرسش (به فارسی):", "پاسخ (به فارسی):")},
    "Mandarin": {"remove_pattern": r"\(in Mandarin\)", "alt_extraction": ("问题（中文）：", None)},
    "Korean": {"remove_pattern": r"\(in Korean\)", "alt_extraction": ("질문 (한국어):", "답변 (한국어):")},
    "French": {
        "remove_pattern": r"\(in French\)",
        # For French, several alternative markers might be needed.
        "alt_extraction": None,
        "prefix_removals": ["Question (in French) :", "Question (in French) :  "],
    },
    "Bengali": {"remove_pattern": r"\(in Bengali\)", "alt_extraction": ("প্রশ্ন (বাংলায়):", None)},
}


def process_language(input_file: str, output_file: str, language: str) -> None:  # noqa: PLR0912
    """
    Extract and clean question/answer pairs from a JSON file for a specified language.

    Loads data from input_file, processes "Content" or "content" fields to extract questions
    and answers using markers ("Question", "Answer") and language-specific rules from
    LANGUAGE_CONFIG, and saves the cleaned data to output_file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
        language (str): Target language (e.g., "Urdu", "Spanish").

    Returns
    -------
        None
    """
    # Load configuration for the target language; if not defined, use default (empty config).
    config: dict[str, Any] = LANGUAGE_CONFIG.get(language, {"remove_pattern": "", "alt_extraction": None})
    remove_pattern = config.get("remove_pattern", "")
    alt_extraction = config.get("alt_extraction")
    prefix_removals = config.get("prefix_removals", [])

    # Load input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []

    for entry in data:
        # Get the content from either "Content" or "content"
        content = entry.get("Content", "") or entry.get("content", "")
        question = None
        answer = None

        # Primary extraction: if "Question" is in content, split on it and then split on "Answer"
        if "Question" in content:
            extracted = content.split("Question", 1)[1].strip()
            if "Answer" in extracted:
                question, answer = extracted.split("Answer", 1)
                question = question.strip()
                answer = answer.strip()
            else:
                question = extracted

        # If no question was extracted (or it's empty), try alternative extraction if provided.
        if (not question or question == "") and alt_extraction:
            q_marker, a_marker = alt_extraction
            if q_marker in content:
                question = content.split(q_marker, 1)[1].strip()
                if a_marker and a_marker in question:
                    question, answer = question.split(a_marker, 1)
                    question = question.strip()
                    answer = answer.strip()

        # Remove language-specific pattern from question and answer, if they exist.
        if question:
            question = re.sub(remove_pattern, "", question).strip()
        if answer:
            answer = re.sub(remove_pattern, "", answer).strip()

        # Remove any prefixes specified in config (for example in French)
        for prefix in prefix_removals:
            if question and question.startswith(prefix):
                question = question.replace(prefix, "", 1).strip()

        # Clean up: remove trailing and leading colons and whitespace
        if question:
            question = question.strip().strip(":")
            # Remove extra quotes if present
            question = question.replace('"', "").strip()
            # Remove any leading/trailing smart quotes
            question = question.strip("“”")
        if answer:
            answer = answer.strip().strip(":")
            answer = answer.replace('"', "").strip()
            answer = answer.strip("“”")

        # Prepare processed entry
        processed_entry = {
            "ID": entry.get("ID"),
            "Attribute": entry.get("Attribute"),
            "Content": content,
            "Question": question,
            "Answer": answer,
        }
        processed_data.append(processed_entry)

    # Save processed data to output file in UTF-8 encoding
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f"Processed data saved to {output_file}")


def combine_json_data(output_dir: str, selected_file: str, base_file: str, output_file: str, language: str) -> None:
    """
    Combine language-specific data with base English data.

    This function loads a language-specific JSON file (e.g., Bengali) and a base
    English JSON file, then combines entries by matching on 'ID' and 'Attribute'.
    If a match is found, it creates a combined entry that includes both English and
    language-specific questions and answers, and saves the result to an output file.

    Args:
        output_dir (str): Directory containing language JSON files (not used in this function).
        selected_file (str): Path to the language-specific JSON file.
        base_file (str): Path to the base English JSON file.
        output_file (str): Path where the combined JSON data will be saved.
        language (str): Language identifier for the language-specific data.
    """
    # Load language-specific data
    with open(selected_file, "r", encoding="utf-8") as f:
        selected_data = json.load(f)
    print(f"Number of selected entries (language-specific) from {selected_file}: {len(selected_data)}")

    # Load base English data
    with open(base_file, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    print(f"Number of entries in base file (English data) from {base_file}: {len(base_data)}")

    combined_data = []

    # Iterate over each entry in the base file
    for english_entry in base_data:
        entry_id = english_entry["ID"]
        entry_attribute = english_entry["Attribute"]

        # Search for a matching entry in the language-specific file
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
                "Answer(English)": english_entry["Answer"],
                "Language": language,
                f"Question({language})": matching_entry["Question"],
                f"Answer({language})": matching_entry["Answer"],
            }
            combined_data.append(combined_entry)
        else:
            print(f"No match found for ID: {entry_id}, Attribute: {entry_attribute} in the selected file")

    # Save the combined data to the output JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"Combined data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and combine JSONL and JSON files.")

    parser.add_argument("--jsonl_dir", required=True, help="Directory containing .jsonl files to process.")
    parser.add_argument(
        "--output_dir", required=True, help="Directory where processed and combined files will be saved."
    )
    parser.add_argument(
        "--input_folder", required=True, help="Directory containing language-specific JSON files for combining."
    )
    parser.add_argument("--base_file", required=True, help="Path to the base English JSON file.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process JSONL files
    jsonl_files = glob.glob(os.path.join(args.jsonl_dir, "*.jsonl"))
    output_files = [
        os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(file))[0]}_processed.json")
        for file in jsonl_files
    ]

    for input_file, output_file in zip(jsonl_files, output_files):
        process_jsonl_files([input_file], output_file)

    # Process language-specific JSON files
    languages = [
        "Urdu",
        "Tamil",
        "Spanish",
        "Punjabi",
        "Portuguese",
        "Persian",
        "Mandarin",
        "Korean",
        "French",
        "Bengali",
    ]
    for language in languages:
        input_file = os.path.join(args.output_dir, f"results_{language}_processed.json")
        output_file = os.path.join(args.output_dir, f"results_{language}_processed_output.json")
        process_language(input_file, output_file, language)
    print("Language-specific files processed.")

    # Combine processed language files with base file
    input_files = os.listdir(args.input_folder)

    for selected_file in input_files:
        language = selected_file.split("_")[1]  # You might want to make this more robust if needed
        selected_file_path = os.path.join(args.input_folder, selected_file)
        output_file = os.path.join(args.output_dir, f"Eval2_{language}_combined.json")
        combine_json_data(args.output_dir, selected_file_path, args.base_file, output_file, language)

# To run the script,
# python postprocess_eval2.py \
#     --jsonl_dir <path_to_jsonl_files> \
#     --output_dir <path_to_output_directory> \
#     --input_folder <path_to_language_json_files> \
#     --base_file <path_to_base_english_json_file>

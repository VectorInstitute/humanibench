"""HumanI Bench — task 7: postprocess VLM result JSON files."""

import json
import os
import re
from typing import Any


# Common processing logic for all models
def process_data(data: list[dict[str, Any]], model_config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Process the data according to model-specific rules.

    Args:
        data (list): List of dictionaries containing the data to be processed.
        model_config (dict): Dictionary containing model-specific processing rules.

    Returns
    -------
        list: List of dictionaries containing the cleaned data.
    """
    results = []
    for entry in data:
        predicted_answer: str = entry["Predicted_Answer"]
        reasoning: str | None = None

        # Apply model-specific processing rules
        if model_config.get("reasoning_split"):
            if model_config["reasoning_split"] in predicted_answer:
                predicted_answer, reasoning = predicted_answer.split(model_config["reasoning_split"])
            reasoning = reasoning.strip() if reasoning else None

        # Apply answer extraction rule
        if model_config.get("answer_split"):
            predicted_answer = predicted_answer.split(model_config["answer_split"])[-1]

        # Apply any additional text clean up
        pa = clean_text(predicted_answer, model_config.get("cleanup_tags"))
        predicted_answer = "" if pa is None else pa
        if reasoning:
            reas = clean_text(reasoning, model_config.get("cleanup_tags"))
            reasoning = reas if reas is not None else None

        # Add cleaned data to results
        results.append(
            {
                "ID": entry["ID"],
                "Question": entry["Question"],
                "Predicted_Answer": entry["Predicted_Answer"],
                "Model_Answer": predicted_answer.strip(),
                "Model_Reasoning": reasoning,
                "Ground_Truth": entry["Ground_Truth"],
                "Attribute": entry["Attribute"],
            }
        )

    return results


# Helper function to clean the text by removing specified tags
def clean_text(text: str | None, tags: list[str] | None = None) -> str | None:
    """
    Clean the text by removing specified tags and whitespace.

    Args:
        text (str): The text to be cleaned.
        tags (list): List of regex patterns to remove from the text.

    Returns
    -------
        str: The cleaned text.
    """
    if not text:
        return None
    if tags:
        for tag in tags:
            text = re.sub(tag, "", text)
    return text.strip()


# Model configurations: Rules to apply for each model
MODEL_CONFIGS = {
    "Llama_Vision": {
        "reasoning_split": "Reasoning:",
        "answer_split": "Answer:",
        "cleanup_tags": [r"<reasoning>", r"<answer>", r"[<>]"],
    },
    "Phi": {
        "reasoning_split": "Reasoning:",
        "answer_split": "Answer:",
        "cleanup_tags": [r"<reasoning>", r"<answer>", r"[<>]"],
    },
    "Aya": {
        "reasoning_split": "Reasoning:",
        "answer_split": "Answer:",
        "cleanup_tags": [r"<reasoning>", r"<answer>", r"[<>]"],
    },
    "gemma3_12b": {
        "reasoning_split": "Reasoning:",
        "answer_split": "Answer:",
        "cleanup_tags": [r"<reasoning>", r"<answer>", r"[<>]"],
    },
}


# Main function to process files
def process_files(results_folder: str, save_folder: str) -> None:
    """Process JSON result files, apply model rules, and save under save_folder."""
    os.makedirs(save_folder, exist_ok=True)
    files = [f for f in os.listdir(results_folder) if f.endswith(".json")]

    for file in files:
        cleaned_file_path = os.path.join(save_folder, file.replace(".json", "_cleaned.json"))
        with open(os.path.join(results_folder, file)) as f:
            data = json.load(f)

        # Identify model from the filename
        model_name = next((model for model in MODEL_CONFIGS if model in file), None)
        if model_name:
            model_config = MODEL_CONFIGS[model_name]
            cleaned_data = process_data(data, model_config)
            # Save cleaned data
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=4)
            print(f"File {file} cleaned and saved to {cleaned_file_path}")
        else:
            print(f"File {file} not processed due to unknown model")


if __name__ == "__main__":
    results_folder = "./results"
    save_folder = "./results/cleaned"
    process_files(results_folder, save_folder)

# To run the script:
# python postprocess.py

"""HumanI Bench — task 2 (instance identity): postprocess."""

import json
import os
import re
from argparse import ArgumentParser
from typing import Any


def clean_prediction(predicted_answer: Any, model_type: str) -> tuple[str, str | None]:
    """Clean a predicted answer by extracting the answer and reasoning parts."""
    # If predicted_answer is a list, take its first element.
    if isinstance(predicted_answer, list):
        predicted_answer = predicted_answer[0]

    # Model-specific pre-cleaning:
    if model_type.lower() == "phi":
        predicted_answer = re.sub(r"ASSISTANT:\s*", "", predicted_answer)

    # Split on "Reasoning:" if present.
    if "Reasoning:" in predicted_answer:
        answer_part, reasoning = predicted_answer.split("Reasoning:", 1)
    else:
        answer_part, reasoning = predicted_answer, None

    # If "Answer:" is present, take only the portion after it.
    if "Answer:" in answer_part:
        answer_part = answer_part.split("Answer:", 1)[1]

    # Strip whitespace.
    answer_part = answer_part.strip()
    if reasoning is not None:
        reasoning = reasoning.strip()

    # Remove angle brackets.
    answer_part = re.sub(r"[<>]", "", answer_part)
    if reasoning:
        reasoning = re.sub(r"[<>]", "", reasoning)

    return answer_part, reasoning


def process_predictions(data: list[Any], model_type: str) -> list[Any]:
    """Process a list of prediction dictionaries and return cleaned results."""
    results = []
    for entry in data:
        raw_pred = entry.get("Predicted_Answer")
        clean_ans, clean_reasoning = clean_prediction(raw_pred, model_type)
        results.append(
            {
                "ID": entry.get("ID"),
                "Question": entry.get("Question"),
                "Predicted_Answer": raw_pred,
                "Model_Answer": clean_ans,
                "Model_Reasoning": clean_reasoning,
                "Ground_Truth": entry.get("Ground_Truth"),
                "Attribute": entry.get("Attribute"),
            }
        )
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Clean prediction results from a JSON file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument(
        "--model_type", type=str, default="Phi", help="Model type (e.g. Phi, llava, deepseek, janus, molmo, etc.)"
    )
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        cleaned_data = process_predictions(data, args.model_type)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
        print(f"Cleaned data saved to {args.output_file}")
    else:
        print(f"Input file {args.input_file} not found.")

# To run this script:
# python postprocess.py \
# --input_file <path_to_input_json> \
# --output_file <path_to_output_json> \
# --model_type <model_type>

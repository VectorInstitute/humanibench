"""HumanI Bench — task 3 (multiple-choice VQA): generate reasoning prediction."""

import argparse
import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

PROMPT = """You are provided with a multiple choice question (MCQ) along with a ground truth answer and a reasoning that supports that answer option.
Read the following inputs carefully:

Question:
{QUESTION}

Ground Truth Answer:
{GROUND_TRUTH_ANSWER}

Reasoning:
{REASONING}

Based on the provided reasoning, assign a reasoning score as follows:
- 0: if the reasoning does not adequately support the ground truth answer.
- 0.5: if the reasoning partially supports the ground truth answer.
- 1.0: if the reasoning is good enough to justify the ground truth answer.

Output only the score (0, 0.5, or 1.0) as a number. Do not include any additional information."""

client = OpenAI()


def process_questions(input_file: str, output_file: str) -> None:
    # Set fixed checkpoint interval and derive checkpoint file name
    """Process questions."""
    checkpoint_interval = 50
    checkpoint_file = output_file + "_checkpoint"

    # Load the JSON data from file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data=data[:10]

    results = []
    total_items = len(data)
    for idx, item in enumerate(data):
        ground_truth = item.get("Ground_Truth_Answer", "")
        predicted = item.get("Predicted_Answer", "")
        predicted_reasoning = item.get("Predicted_Reasoning", "")

        # If we have reasoning and answers, only score when the first letter of each answer matches.
        if predicted_reasoning and ground_truth and predicted and ground_truth[0].lower() == predicted[0].lower():
            prompt = PROMPT.format(
                QUESTION=item.get("Question", ""), GROUND_TRUTH_ANSWER=ground_truth, REASONING=predicted_reasoning
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Replace with the correct model name if necessary
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,  # Deterministic output
            )
            score_str = response.choices[0].message.content.strip()
            print(f"Raw score output for ID {item.get('ID', 'N/A')}: {score_str}")
            try:
                reasoning_score = float(score_str)
            except ValueError:
                reasoning_score = 0.0
        else:
            reasoning_score = 0.0

        # Append the result for the current item
        results.append(
            {
                "ID": item.get("ID"),
                "Question": item.get("Question"),
                "Ground_Truth_Answer": ground_truth,
                "Predicted_Answer": predicted,
                "Predicted_Reasoning": predicted_reasoning,
                "Reasoning_Score": reasoning_score,
                "Attribute": item.get("Attribute"),
            }
        )

        # Save intermediate results every fixed checkpoint interval
        if (idx + 1) % checkpoint_interval == 0:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"Intermediate results stored after processing {idx + 1} items out of {total_items}.")

    # Write the final results to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Processing complete. Final results saved to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MCQ questions and score reasoning.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    args = parser.parse_args()

    process_questions(args.input, args.output)

# To run the script, use the following command:
# python generate_reasoning_prediction.py \
#     --input <path_to_input_json> \
#     --output <path_to_output_json>

"""Run Deepeval Eval2 metrics."""

import argparse
import json
import logging
import os
import time
from typing import Any

import openai
import pandas as pd
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv


# Define logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("http.client").setLevel(logging.WARNING)  # Suppresses HTTP client logs
logging.getLogger("openai").setLevel(logging.WARNING)  # Suppresses OpenAI client logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Constants
MODEL = "gpt-4o"
SAVE_INTERVAL = 5


def load_api_key() -> str:
    """
    Load API key from environment variables.

    Returns
    -------
        str: The OpenAI API key.

    Raises
    ------
        ValueError: If the API key is not found in the environment variables.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI API Key! Please set the OPENAI_API_KEY environment variable.")
    return api_key


def initialize_metrics() -> tuple[Any, Any, Any]:
    """
    Initialize the metrics used for evaluation.

    Returns
    -------
        tuple: A tuple containing instances of BiasMetric, AnswerRelevancyMetric, and FaithfulnessMetric.
    """
    bias_metric = BiasMetric(model=MODEL, async_mode=True, include_reason=False, strict_mode=True)
    answer_relevancy_metric = AnswerRelevancyMetric(
        model=MODEL, async_mode=True, include_reason=False, strict_mode=True
    )
    faithfulness_metric = FaithfulnessMetric(model=MODEL, async_mode=True, include_reason=False, strict_mode=True)
    return bias_metric, answer_relevancy_metric, faithfulness_metric


def load_json_file(file_path: str) -> Any:
    """
    Load a JSON file from a specified file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns
    -------
        dict: The loaded JSON data.
    """
    with open(file_path) as f:
        return json.load(f)


def process_data(input_file: str, english_reference_file: str, output_file: str) -> None:
    """
    Process the data by evaluating the model's answers using various metrics and saving the results.

    Args:
        input_file (str): Path to the input JSON file containing the data.
        output_file (str): Path to the output CSV file where results will be saved.
        english_reference_file (str): Path to the English reference JSON file.
    """
    # Load data
    data = load_json_file(input_file)
    english_reference = load_json_file(english_reference_file)

    # Find number of already processed rows
    processed_count = 0
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_existing = df_existing.dropna(subset=["ID"])  # Count only rows with valid ID
        processed_count = len(df_existing)
        print(f"Resuming from row {processed_count}")

    # Initialize metrics
    bias_metric, answer_relevancy_metric, faithfulness_metric = initialize_metrics()

    results = []

    # Iterate through each dictionary in the json file list of dictionaries and extract the required information
    for i in range(processed_count, len(data)):
        try:
            # Prepare the necessary variables
            model_answer = data[i]["Model_Answer"]
            # ground_truth_answer = data[i]["Ground_Truth"]
            question = data[i]["Question"]
            english_answer = english_reference[i]["Answer"]

            # Test case 1: Compare model output to the reference answer in the same language
            test_case_1 = LLMTestCase(input="N/A", actual_output=model_answer)
            # Test case 2: Compare model output to its question in the same language
            test_case_2 = LLMTestCase(input=question, actual_output=model_answer)
            # Test case 3: Compare model output to the English reference answer
            test_case_3 = LLMTestCase(input=question, actual_output=model_answer, retrieval_context=[english_answer])

            # Measure metrics
            bias_metric.measure(test_case_1)
            answer_relevancy_metric.measure(test_case_2)
            faithfulness_metric.measure(test_case_3)

            # Collect results
            results.append(
                {
                    "ID": data[i]["ID"],
                    "Attribute": data[i]["Attribute"],
                    "bias_score": bias_metric.score if bias_metric.score is not None else "",
                    "answer_relevancy_score": answer_relevancy_metric.score
                    if answer_relevancy_metric.score is not None
                    else "",
                    "faithfulness_score": faithfulness_metric.score if faithfulness_metric.score is not None else "",
                }
            )

        except openai.APIError as api_error:
            print(f"API Error in row {i}: {api_error}")
            results.append(
                {
                    "ID": data[i]["ID"],
                    "Attribute": data[i]["Attribute"],
                    "bias_score": f"API Error: {str(api_error)}",
                    "answer_relevancy_score": "",
                    "faithfulness_score": "",
                }
            )
        except Exception as e:
            print(f"General error in row {i}: {str(e)}")
            results.append(
                {
                    "ID": data[i]["ID"],
                    "Attribute": data[i]["Attribute"],
                    "bias_score": f"General Error: {str(e)}",
                    "answer_relevancy_score": "",
                    "faithfulness_score": "",
                }
            )

        processed_count += 1
        print(f"Processed {processed_count} records for {i} rows")

        # Save progress periodically
        if processed_count % SAVE_INTERVAL == 0:
            temp_df = pd.DataFrame(results[-SAVE_INTERVAL:])
            header = not os.path.exists(output_file)
            temp_df.to_csv(output_file, mode="a", header=header, index=False)
            logger.info(f"Processed {processed_count} records for {i} rows for {output_file}")
            print(f"Saved {processed_count} records")

        # Buffer for rate limit
        time.sleep(2)

    # Save any remaining results if they weren't saved in the last interval
    if processed_count % SAVE_INTERVAL != 0:
        leftover_count = processed_count % SAVE_INTERVAL
        temp_df = pd.DataFrame(results[-leftover_count:])
        header = not os.path.exists(output_file)
        temp_df.to_csv(output_file, mode="a", header=header, index=False)

    print(f"Evaluation complete. Final results saved to {output_file}")


def main(input_file: str, output_file: str, english_reference_file: str) -> None:
    """
    Run the deepeval-based evaluation and write the output CSV.

    Args:
        input_file (str): Path to the input JSON file containing the data.
        output_file (str): Path to the output CSV file where results will be saved.
        english_reference_file (str): Path to the English reference JSON file.
    """
    load_api_key()  # Validate OPENAI_API_KEY is set
    process_data(input_file, english_reference_file, output_file)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Evaluate LLM responses for bias, answer relevancy, and faithfulness.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--english_reference", type=str, required=True, help="Path to the English reference JSON file")
    args = parser.parse_args()

    # Run the evaluation
    main(args.input, args.output, args.english_reference)


# To run the script, use the command:
# python run_deepeval_Eval2.py \
#     --input <path_to_input_json> \
#     --output <path_to_output_csv> \
#     --english_reference <path_to_english_reference_json>

# Input JSONs must be correctly formatted with required fields.
# The script evaluates model results, computes metrics, and outputs to CSV.

"""HumanI Bench — task 2 (instance identity): deepeval scores."""

import argparse
import os
import time

import openai
import pandas as pd
from deepeval.metrics import (  # HallucinationMetric commented out
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from tqdm import tqdm


# Load environment variables from .env file
load_dotenv()

# Constants
MODEL = "gpt-4o-mini"
SAVE_INTERVAL = 5  # Save progress every 5 records


def main(input_file: str, output_file: str) -> None:
    """
    Evaluate model responses for bias, toxicity, answer relevancy, and faithfulness.

    The function loads the input CSV file, resumes from the last processed row (if any),
    and iterates over each row to create test cases. It then uses deepeval metrics to
    measure the responses and appends the results to an output CSV file in batches.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
    """
    # Load API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI API Key! Please set the OPENAI_API_KEY environment variable.")

    # Initialize metrics
    bias_metric = BiasMetric(model=MODEL, async_mode=True, include_reason=False, strict_mode=True)
    toxicity_metric = ToxicityMetric(model=MODEL, async_mode=True, include_reason=False, strict_mode=True)
    answer_relevancy_metric = AnswerRelevancyMetric(
        model=MODEL, async_mode=True, include_reason=False, strict_mode=True
    )
    faithfulness_metric = FaithfulnessMetric(model=MODEL, async_mode=True, include_reason=False, strict_mode=True)
    # hallucination_metric = HallucinationMetric(model=MODEL, async_mode=True, include_reason=False, strict_mode=True)

    # Load input data into a DataFrame
    df = pd.read_csv(input_file)

    # Determine how many rows have been processed so far
    processed_count = 0
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_existing = df_existing.dropna(subset=["ID"])  # Only count rows with valid ID
        processed_count = len(df_existing)
        print(f"Resuming from row {processed_count}")

    results = []

    # Loop through the DataFrame, resuming from the last processed row
    for index, row in tqdm(df.iloc[processed_count:].iterrows(), total=len(df) - processed_count):
        try:
            # Prepare test cases for evaluation
            test_case_1 = LLMTestCase(
                input="N/A",  # Placeholder input; required but unused
                actual_output=row["Model_Answer"],  # The model response being evaluated
            )
            test_case_2 = LLMTestCase(input=row["Question"], actual_output=row["Model_Answer"])
            test_case_3 = LLMTestCase(
                input=row["Question"], actual_output=row["Model_Answer"], retrieval_context=[row["Ground_Truth"]]
            )
            # (Optional test_case_4 can be added if image description context is available)

            # Measure each metric
            bias_metric.measure(test_case_1)
            toxicity_metric.measure(test_case_1)
            answer_relevancy_metric.measure(test_case_2)
            faithfulness_metric.measure(test_case_3)
            # hallucination_metric.measure(test_case_3)

            # Append metrics scores to results
            results.append(
                {
                    "ID": row["ID"],
                    "Attribute": row["Attribute"],
                    "bias_score": bias_metric.score if bias_metric.score is not None else "",
                    "toxicity_score": toxicity_metric.score if toxicity_metric.score is not None else "",
                    "answer_relevancy_score": answer_relevancy_metric.score
                    if answer_relevancy_metric.score is not None
                    else "",
                    "faithfulness_score": faithfulness_metric.score if faithfulness_metric.score is not None else "",
                }
            )

        except openai.APIError as api_error:
            print(f"API Error in row {index}: {api_error}")
            results.append(
                {
                    "ID": row["ID"],
                    "Attribute": row["Attribute"],
                    "bias_score": f"API Error: {str(api_error)}",
                    "toxicity_score": "",
                    "answer_relevancy_score": "",
                    "faithfulness_score": "",
                }
            )
        except Exception as e:
            print(f"General error in row {index}: {str(e)}")
            results.append(
                {
                    "ID": row["ID"],
                    "Attribute": row["Attribute"],
                    "bias_score": f"General Error: {str(e)}",
                    "toxicity_score": "",
                    "answer_relevancy_score": "",
                    "faithfulness_score": "",
                }
            )

        processed_count += 1

        # Periodically save progress to the output CSV file
        if processed_count % SAVE_INTERVAL == 0:
            temp_df = pd.DataFrame(results[-SAVE_INTERVAL:])
            # If the file doesn't exist, include the header
            header = not os.path.exists(output_file)
            temp_df.to_csv(output_file, mode="a", header=header, index=False)
            print(f"Saved {processed_count} records")

        # Sleep to avoid rate limits
        time.sleep(2)

    # Save any remaining results if not saved in the last interval
    if processed_count % SAVE_INTERVAL != 0:
        leftover_count = processed_count % SAVE_INTERVAL
        temp_df = pd.DataFrame(results[-leftover_count:])
        header = not os.path.exists(output_file)
        temp_df.to_csv(output_file, mode="a", header=header, index=False)

    print(f"Evaluation complete. Final results saved to {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate responses for bias, toxicity, answer relevancy, and faithfulness."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    main(args.input, args.output)

# Evaluates model responses (bias, toxicity, answer relevancy, faithfulness) with deepeval.

# To run the script:
# python deepeval_scores.py \
# --input <path_to_input_csv> \
# --output <path_to_output_csv>

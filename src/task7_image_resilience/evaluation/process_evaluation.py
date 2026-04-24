"""HumanI Bench — task 7 (image resilience): process evaluation."""

import argparse
import json
import os
from typing import Any

import pandas as pd


def extract_scores_and_matches(directory_path: str) -> Any:
    # Prepare a list to store the results for all files
    """Extract scores and matches."""
    summary_data = []

    # Iterate through all .json files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):  # Assuming processed files have '_processed.json'
            # Build the full file path
            file_path = os.path.join(directory_path, filename)

            # Read the JSON data from the file
            with open(file_path, "r") as file:
                data = json.load(file)

            # Extract scores and matches from the data
            total_matches = 0
            total_score = 0
            for entry in data:
                # Count the number of matches and sum the scores
                if entry.get("match") == "Yes":
                    total_matches += 1
                total_score += entry.get("score", 0)

            # Calculate the average score
            average_score = total_score / len(data) if len(data) > 0 else 0

            # Add the results for the file to the summary data
            summary_data.append({"file_name": filename, "matches": total_matches, "average_score": average_score})

    # Convert the summary data to a Pandas DataFrame
    df = pd.DataFrame(summary_data)

    # Optionally, you can save the table to a CSV file
    output_file = os.path.join(directory_path, "summary_table.csv")
    df.to_csv(output_file, index=False)

    # Display the table
    print(df)

    return df


if __name__ == "__main__":
    # Take the directory path as a command line argument
    parser = argparse.ArgumentParser(description="Extract scores and matches from processed evaluation files.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing processed evaluation files.")
    args = parser.parse_args()
    directory_path = args.directory_path
    extract_scores_and_matches(directory_path)

# Example usage:
# python process_evaluation.py \
# --directory_path <path_to_your_directory_containing_json_files>

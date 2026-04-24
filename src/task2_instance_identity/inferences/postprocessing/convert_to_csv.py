"""HumanI Bench — task 2 (instance identity): convert to csv."""

import argparse
import json
import os

import pandas as pd


def convert_json_to_csv(input_folder: str, output_folder: str) -> None:
    """
    Convert all JSON files in the input folder to CSV files saved in the output folder.

    Each JSON file should be a list of dictionaries with the columns:
    "ID", "Question", "Predicted_Answer", "Model_Answer", "Model_Reasoning", "Ground_Truth", "Attribute".
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(input_folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Processing {file}: {len(data)} entries found.")
            df = pd.DataFrame(
                data,
                columns=[
                    "ID",
                    "Question",
                    "Predicted_Answer",
                    "Model_Answer",
                    "Model_Reasoning",
                    "Ground_Truth",
                    "Attribute",
                ],
            )
            output_file = os.path.join(output_folder, file.replace(".json", ".csv"))
            df.to_csv(output_file, index=False)
            print(f"Saved: {file.replace('.json', '.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON files to CSV.")
    parser.add_argument("input_folder", type=str, help="Input folder containing JSON files.")
    parser.add_argument("output_folder", type=str, help="Output folder to save CSV files.")
    args = parser.parse_args()
    convert_json_to_csv(args.input_folder, args.output_folder)

# To run this script, use the command line:
# python convert_to_csv.py <input_folder> <output_folder>

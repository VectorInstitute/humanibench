"""HumanI Bench — task 6 (empathetic captioning): to csv."""

import argparse
import csv
import json
from argparse import Namespace
from typing import Any, Sequence, cast


def load_json_data(json_filename: str) -> list[dict[str, Any]]:
    """Load data from a JSON file."""
    with open(json_filename, "r", encoding="utf-8") as file:
        return cast(list[dict[str, Any]], json.load(file))


def write_to_csv(data: list[dict[str, Any]], csv_filename: str, fieldnames: Sequence[str]) -> None:
    """Write data to a CSV file."""
    with open(csv_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            row = {
                "id": item.get("id", ""),
                "image_path": item.get("image_path", ""),
                "csv_simple": item.get("csv_captions", {}).get("csv_simple", ""),
                "csv_emphatic": item.get("csv_captions", {}).get("csv_emphatic", ""),
                "model_simple": item.get("model_captions", {}).get("model_simple", ""),
                "model_empathetic": item.get("model_captions", {}).get("model_empathetic", ""),
            }
            writer.writerow(row)


def main(args: Namespace) -> None:
    # Define column names for CSV
    """Run the main workflow."""
    fieldnames = ["id", "image_path", "csv_simple", "csv_emphatic", "model_simple", "model_empathetic"]

    # Load the JSON data and write it to a CSV file
    data = load_json_data(args.json_filename)
    write_to_csv(data, args.csv_filename, fieldnames)

    print(f"CSV file created at: {args.csv_filename}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert JSON data to CSV format.")
    parser.add_argument("--json_filename", type=str, default="data.json", help="Path to the input JSON file")
    parser.add_argument("--csv_filename", type=str, default="output.csv", help="Path to the output CSV file")

    args = parser.parse_args()
    main(args)

# This script converts a JSON file containing image paths and captions into a CSV file.
# It extracts relevant fields and writes them into a structured CSV format.

# To run the script, use the following command:
# python to-csv.py \
#     --json_filename <path_to_input_json> \
#     --csv_filename <path_to_output_csv>

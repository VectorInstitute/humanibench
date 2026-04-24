"""HumanI Bench — task 4 (multilingual): get stats deepeval."""

import os
from typing import Any

import pandas as pd


def calculate_percent(col_0: Any, col_1: Any) -> Any:
    """
    Calculate the percentage of column_1 relative to the total of column_0 and column_1.

    Returns
    -------
        float: The percentage of column_1 values out of the total.
    """
    total = col_0 + col_1
    return (col_1 / total * 100) if total > 0 else 0


def process_file(file_path: str, stats_columns: Any, filter_attribute: Any = None) -> Any:
    """
    Process a single CSV file and gather statistics for relevant columns.

    Returns
    -------
        list: List of statistics computed for the file.
    """
    df = pd.read_csv(file_path)

    # Filter by attribute if needed
    if filter_attribute:
        df = df[df["Attribute"] == filter_attribute]

    stats: list[Any] = [os.path.basename(file_path)]  # Add the file name to the stats list

    # Initialize variables for each column
    bias_0 = bias_1 = answer_relevancy_0 = answer_relevancy_1 = faithfulness_0 = faithfulness_1 = 0

    # Collect counts for each relevant column
    for column in stats_columns:
        if column not in {"ID", "Attribute"}:
            column_0 = df[column].value_counts().get(0, 0)
            column_1 = df[column].value_counts().get(1, 0)

            stats.append(column_0)
            stats.append(column_1)

            # Add to the respective counters
            if column == "bias_score":
                bias_0, bias_1 = column_0, column_1
            elif column == "answer_relevancy_score":
                answer_relevancy_0, answer_relevancy_1 = column_0, column_1
            elif column == "faithfulness_score":
                faithfulness_0, faithfulness_1 = column_0, column_1

    # Calculate percentages
    bias_1_percent = round(calculate_percent(bias_0, bias_1), 2)
    answer_relevancy_1_percent = round(calculate_percent(answer_relevancy_0, answer_relevancy_1), 2)
    faithfulness_1_percent = round(calculate_percent(faithfulness_0, faithfulness_1), 2)

    # Append the percentages to the stats
    stats.extend([bias_1_percent, answer_relevancy_1_percent, faithfulness_1_percent])

    # Add the total number of rows in this file
    stats.append(len(df))

    return stats


def get_stats(folder_path: str, output_filename: str = "stats.csv") -> None:
    """Collect statistics from CSV files in the provided folder."""
    files = os.listdir(folder_path)
    output = []

    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)

            # Process the file and gather statistics
            stats = process_file(
                file_path, stats_columns=["bias_score", "answer_relevancy_score", "faithfulness_score"]
            )

            # Append the results
            output.append(stats)

    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(
        output,
        columns=[
            "File Name",
            "bias_score_0",
            "bias_score_1",
            "answer_relevancy_score_0",
            "answer_relevancy_score_1",
            "faithfulness_score_0",
            "faithfulness_score_1",
            "bias_1_percent",
            "answer_relevancy_1_percent",
            "faithfulness_1_percent",
            "Total Rows",
        ],
    )
    output_df.to_csv(output_filename, index=False)


def get_stats_attributes(folder_path: str, output_filename: str = "stats_attributes.csv") -> None:
    """Collect attribute-based statistics from CSV files in a folder."""
    files = os.listdir(folder_path)
    output = []
    attributes = ["Gender", "Age", "Ethnicity", "Occupation", "Sport"]

    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)

            for attribute in attributes:
                # Process the file and gather statistics for the current attribute
                stats = process_file(
                    file_path,
                    stats_columns=["bias_score", "answer_relevancy_score", "faithfulness_score"],
                    filter_attribute=attribute,
                )

                # Add the attribute to the stats
                stats.append(attribute)

                # Append the results
                output.append(stats)

    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(
        output,
        columns=[
            "File Name",
            "bias_1_percent",
            "answer_relevancy_1_percent",
            "faithfulness_1_percent",
            "Total Rows",
            "Attribute",
        ],
    )
    output_df.to_csv(output_filename, index=False)


# Example usage:
folder_path = "./Eval2_deepeval"  # Path to the folder containing CSV files

# Get overall stats
get_stats(folder_path)

# Get stats for attributes
get_stats_attributes(folder_path)

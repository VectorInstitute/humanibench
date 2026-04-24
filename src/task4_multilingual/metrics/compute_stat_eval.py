"""Compute evaluation metrics (accuracy, F1, etc.)."""

import argparse
import os
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def extract_options(input_json: str, answer_column: str, new_column: str) -> Any:
    """
    Extract the answer option (first character) from a dataset's answer column.

    Supports multiple language options.

    Returns
    -------
        pd.DataFrame: DataFrame with an additional column for extracted answer options.
    """
    df = pd.read_json(input_json)

    def extract_answer_option(answer: str) -> str | None:
        if answer:
            valid_options = ["A", "B", "C", "D", "ক", "খ", "গ", "ঘ", "الف", "ب", "ج", "د"]
            first_char = answer[0]
            return first_char if first_char in valid_options else None
        return None

    df[new_column] = df[answer_column].astype(str).apply(extract_answer_option)

    # Debugging: Print rows with missing or empty IDs
    print(f"Rows with missing IDs in {input_json}:")
    print(df[df["ID"].isna()])

    return df


def process_all_models(result_folder: str, eval3_dataset_folder: str) -> Any:
    """
    Process results for all models, compute accuracy, precision, recall, and F1 scores.

    Returns
    -------
        pd.DataFrame: A DataFrame with computed metrics for each model.
    """
    all_results = []

    for filename in os.listdir(result_folder):
        if filename.endswith(".json"):
            model_name = filename.replace(".json", "")
            model_path = os.path.join(result_folder, filename)

            lang = model_name.split("_")[-1]
            eval3_dataset = os.path.join(eval3_dataset_folder, f"Eval3_{lang}.json")

            print(f"Processing model: {model_name}, Language: {lang}")
            print(f"Eval3 Dataset: {eval3_dataset}, File Path: {model_path}")

            eval3_dataset_df = extract_options(eval3_dataset, f"Answer({lang})", "Ground_Label")
            results_df = extract_options(model_path, "Predicted_Answer", "Predicted")

            # Check for missing or empty predictions
            if results_df["Predicted"].isna().any() or (results_df["Predicted"] == "").any():
                print("Missing values found in predictions.")

            # Merge datasets on ID and Attribute
            merged_df = eval3_dataset_df.merge(results_df, on=["ID", "Attribute"], how="inner")
            merged_size = merged_df.shape[0]
            print(f"Merged dataset size for {model_name}: {merged_size} rows")

            # Check if merged size matches expected size
            expected_size = 1268
            if merged_size != expected_size:
                print(
                    f"Warning: Merged dataset for {model_name} has {merged_size} rows, expected {expected_size}. Proceeding anyway."
                )

            # Check for missing values in merged dataset
            missing_after_merge = merged_df.isna().sum()
            print(f"Missing values in merged dataset: {missing_after_merge}")

            # Define ground truth and predicted values
            y_true = merged_df["Ground_Label"]
            y_pred = merged_df["Predicted"]

            # Handle missing predictions and ground truth
            y_pred_full = y_pred.fillna("Missing").replace("", "Missing")
            y_true_full = y_true.fillna("Missing").replace("", "Missing")

            # Exclude specific labels and missing values (based on examples)
            exclude_labels = ["الف. غیر قابل تعیین", "Missing", "unknown"]
            valid_mask = (
                (~y_true_full.isin(exclude_labels))
                & (~y_pred_full.isin(exclude_labels))
                & y_true_full.notna()
                & y_pred_full.notna()
            )
            y_true_valid = y_true_full[valid_mask]
            y_pred_valid = y_pred_full[valid_mask]

            # Calculate metrics
            accuracy = round(accuracy_score(y_true_valid, y_pred_valid), 2)
            precision = round(precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=1), 2)
            recall = round(recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=1), 2)
            f1 = round(f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=1), 2)

            model_results = {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }

            # Process category-wise metrics
            categories = ["Gender", "Age", "Occupation", "Ethnicity", "Sport"]
            for category in categories:
                y_true_cat = merged_df.loc[merged_df["Attribute"] == category, "Ground_Label"]
                y_pred_cat = merged_df.loc[merged_df["Attribute"] == category, "Predicted"]

                if not y_true_cat.empty and not y_pred_cat.empty:
                    valid_mask_cat = (
                        (~y_true_cat.isin(exclude_labels))
                        & (~y_pred_cat.isin(exclude_labels))
                        & y_true_cat.notna()
                        & y_pred_cat.notna()
                    )
                    y_true_cat_valid = y_true_cat[valid_mask_cat]
                    y_pred_cat_valid = y_pred_cat[valid_mask_cat]

                    cat_accuracy = round(accuracy_score(y_true_cat_valid, y_pred_cat_valid), 2)
                    cat_precision = round(
                        precision_score(y_true_cat_valid, y_pred_cat_valid, average="macro", zero_division=1), 2
                    )
                    cat_recall = round(
                        recall_score(y_true_cat_valid, y_pred_cat_valid, average="macro", zero_division=1), 2
                    )
                    cat_f1 = round(f1_score(y_true_cat_valid, y_pred_cat_valid, average="macro", zero_division=1), 2)

                    model_results[f"{category} Accuracy"] = cat_accuracy
                    model_results[f"{category} Precision"] = cat_precision
                    model_results[f"{category} Recall"] = cat_recall
                    model_results[f"{category} F1 Score"] = cat_f1

            all_results.append(model_results)

    # Sort results by model name
    all_results = sorted(all_results, key=lambda x: x["Model"])

    results_df = pd.DataFrame(all_results)
    print(results_df)
    return results_df


# # Example usage
# result_folder = "./Eval5/eval3_results_decoded/"
# eval3_dataset_folder = "./Eval5/eval3_datasets"
# results_df = process_all_models(result_folder, eval3_dataset_folder)

# # Save results to CSV
# results_df.to_csv("stat_results_eval5_eval3.csv", index=False)
# print("Results saved to stat_results_eval5_eval3.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation results and compute metrics.")

    parser.add_argument("--result_folder", type=str, help="Path to the folder containing model result JSON files.")
    parser.add_argument(
        "--eval3_dataset_folder", type=str, help="Path to the folder containing Eval3 ground truth datasets."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="stat_results_eval5_MCQs.csv",
        help="Path to save the output CSV file with evaluation metrics.",
    )

    args = parser.parse_args()

    # Run processing
    results_df = process_all_models(args.result_folder, args.eval3_dataset_folder)

    # Save the results
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

# To run the script, use the command:
# python compute_stat_eval.py \
#     --result_folder <path_to_result_folder> \
#     --eval3_dataset_folder <path_to_eval3_dataset_folder> \
#     --output_csv <path_to_output_csv>

# Note: Ensure the paths you provide are correct, and that any
# required files exist in the specified directories.
# This script processes evaluation results from multiple models,
# computes various metrics, and saves the results to a CSV file.
# It is designed to support multiple languages and allows
# options for excluding specific labels from evaluation.

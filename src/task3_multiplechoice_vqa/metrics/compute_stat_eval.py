"""Computes the Statistic Score for MCQ - Accuracy, Precision, Recall and F1 Score."""

import argparse
import os
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def extract_options(input_json: str, answer_column: str, new_column: str) -> Any:
    """Load dataset and extract first letter of the answer if it's A, B, C, or D."""
    df = pd.read_json(input_json)
    df[new_column] = df[answer_column].astype(str).apply(lambda x: x[0] if x and x[0] in ["A", "B", "C", "D"] else None)
    return df


def process_all_models(result_folder: str, eval3_dataset: str) -> Any:
    """Process all models, compute metrics, and raise errors for missing values, size, or labels."""
    eval3_dataset_df = extract_options(eval3_dataset, "Answer", "Ground_Label")
    all_results = []

    for filename in os.listdir(result_folder):
        if filename.endswith(".json"):
            model_name = filename.replace(".json", "")
            model_path = os.path.join(result_folder, filename)
            print(f"Processing model: {model_name}")

            results_df = extract_options(model_path, "Predicted_Answer", "Predicted")

            if results_df["Predicted"].isna().any() or (results_df["Predicted"] == "").any():
                print("Missing Values found in predictions.")

            # Merging dataframes on ID and Attribute
            merged_df = eval3_dataset_df.merge(results_df, on=["ID", "Attribute"], how="inner")
            merged_size = merged_df.shape[0]
            print(f"Merged dataset size for {model_name}: {merged_size} rows")

            if merged_size != 1844:
                raise ValueError(f"ERROR: Merged dataset for {model_name} has {merged_size} rows, expected 1268.")

            y_true = merged_df["Ground_Label"]
            y_pred = merged_df["Predicted"]

            # Use only non-missing predictions to check if any valid label is missing from predictions
            valid_pred = y_pred.dropna().loc[y_pred != ""]
            unique_true_labels = set(y_true)
            unique_pred_labels = set(valid_pred)
            print(f"- Unique Ground Truth Labels for {model_name}: {unique_true_labels}")
            print(f"- Unique Predicted Labels for {model_name}: {unique_pred_labels}")

            missing_labels = unique_true_labels - unique_pred_labels
            if missing_labels:
                raise ValueError(f"ERROR: Model {model_name} is missing labels: {missing_labels}")

            # For accuracy, count missing predictions as incorrect by replacing them with a placeholder
            y_pred_full = y_pred.fillna("Missing").replace("", "Missing")
            accuracy = accuracy_score(y_true, y_pred_full)

            # For precision, recall, F1, filter out missing predictions
            mask = y_pred.notna() & (y_pred != "")
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            precision = precision_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=1)
            recall = recall_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=1)
            f1 = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=1)

            model_results = {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }

            # Process metrics per category
            categories = ["Gender", "Age", "Occupation", "Ethnicity", "Sport"]
            for category in categories:
                y_true_cat = merged_df.loc[merged_df["Attribute"] == category, "Ground_Label"]
                y_pred_cat = merged_df.loc[merged_df["Attribute"] == category, "Predicted"]

                if not y_true_cat.empty and not y_pred_cat.empty:
                    # Accuracy: count missing predictions as failures.
                    y_pred_cat_full = y_pred_cat.fillna("Missing").replace("", "Missing")
                    cat_accuracy = accuracy_score(y_true_cat, y_pred_cat_full)

                    # For precision, recall, and F1, exclude missing predictions.
                    mask_cat = y_pred_cat.notna() & (y_pred_cat != "")
                    y_true_cat_filtered = y_true_cat[mask_cat]
                    y_pred_cat_filtered = y_pred_cat[mask_cat]
                    cat_precision = precision_score(
                        y_true_cat_filtered, y_pred_cat_filtered, average="macro", zero_division=1
                    )
                    cat_recall = recall_score(
                        y_true_cat_filtered, y_pred_cat_filtered, average="macro", zero_division=1
                    )
                    cat_f1 = f1_score(y_true_cat_filtered, y_pred_cat_filtered, average="macro", zero_division=1)

                    model_results[f"{category} Accuracy"] = cat_accuracy
                    model_results[f"{category} Precision"] = cat_precision
                    model_results[f"{category} Recall"] = cat_recall
                    model_results[f"{category} F1 Score"] = cat_f1

            all_results.append(model_results)

    results_df = pd.DataFrame(all_results)
    print(results_df)
    return results_df


# # Example usage
# parent_folder = "./results"
# result_folder = os.path.join(parent_folder, "eval3")
# eval3_dataset = "./data/eval3/QA_Eval3.json"
# results_df = process_all_models(result_folder, eval3_dataset)

# # Save results to CSV
# results_df.to_csv("stat_results.csv", index=False)
# print("Results saved to stat_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for evaluation results.")
    parser.add_argument("--result_folder", type=str, default="./results/eval3", help="Path to the result folder.")
    parser.add_argument(
        "--eval3_dataset", type=str, default="./data/eval3/QA_Eval3.json", help="Path to the Eval3 dataset."
    )
    parser.add_argument("--output_csv", type=str, default="stat_results.csv", help="Output CSV file name.")
    args = parser.parse_args()
    parent_folder = args.result_folder
    result_folder = os.path.join(parent_folder, "eval3")
    eval3_dataset = args.eval3_dataset
    results_df = process_all_models(result_folder, eval3_dataset)

    # Save results to CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

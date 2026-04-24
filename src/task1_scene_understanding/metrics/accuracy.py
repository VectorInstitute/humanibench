"""HumanI Bench — task 1 (scene understanding): accuracy."""

import argparse

import pandas as pd
import torch
from bert_score import score


def main() -> None:
    """Run the main workflow."""
    parser = argparse.ArgumentParser(description="Consolidation of Results")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to the ground truth CSV file")
    args = parser.parse_args()

    # Load your CSVs
    ground_truth_csv = args.ground_truth
    predicted_csv = args.input

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_gt = pd.read_csv(ground_truth_csv, low_memory=False)
    df_pred = pd.read_csv(predicted_csv, low_memory=False)

    # Merge on common column "id"
    df_merged = df_gt.merge(df_pred, on="id", suffixes=("_gt", "_pred"))

    # Retrieve the responses
    refs = df_merged["response_gt"].fillna("").tolist()
    hyps = df_merged["response_pred"].fillna("").tolist()

    # Calculate BERTScore while forcing the model to use CPU.
    # The `device="cpu"` parameter ensures that the model is not moved to GPU.
    precision, recall, f1 = score(hyps, refs, lang="en", verbose=True, device=device)

    # Add scores to dataframe
    df_merged["BERTScore_P"] = precision.tolist()
    df_merged["BERTScore_R"] = recall.tolist()
    df_merged["BERTScore_F1"] = f1.tolist()

    # Compute the average F1 score across all rows as a semantic similarity measure.
    average_f1 = df_merged["BERTScore_F1"].mean()
    print(f"Average BERTScore F1: {average_f1:.3f}")

    # Optionally, save the merged results with scores for further analysis.
    df_merged.to_csv(args.output, index=False)
    # print("Saved merged results with BERTScore to merged_with_bert_score.csv")


if __name__ == "__main__":
    main()

# To run this script, use the following command:
# python consolidate_results.py \
#     --input <path_to_input_csv> \
#     --output <path_to_output_csv> \
#     --ground_truth <path_to_ground_truth_csv>

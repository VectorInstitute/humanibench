"""HumanI Bench — task 2 (instance identity): eval2 QA generation preprocessing."""

import argparse
import json
import os
from typing import Any

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def load_huggingface_dataset(dataset_name: str) -> Any:
    """Load the Hugging Face dataset."""
    print("Loading Hugging Face dataset...")
    combined_dataset = load_dataset(dataset_name)["train"]
    return pd.DataFrame(combined_dataset)


def load_json_file(json_path: str) -> Any:
    """Load a JSON file."""
    print(f"Loading JSON file from {json_path}...")
    with open(json_path, "r") as file:
        return json.load(file)


def merge_with_images(metadata: Any, dataset_df: Any) -> Any:
    """Merge JSON metadata with the Hugging Face dataset."""
    print("Merging metadata with images...")
    metadata_df = pd.DataFrame.from_dict(metadata, orient="index").reset_index()
    metadata_df.rename(columns={"index": "id"}, inplace=True)

    # Merge with the Hugging Face dataset using `id` and `unique_id`
    merged_df = pd.merge(metadata_df, dataset_df, left_on="id", right_on="unique_id", how="inner")

    if merged_df.empty:
        raise ValueError("No matching data found between metadata and the Hugging Face dataset.")

    return merged_df


def download_and_save_images(merged_df: Any, output_dir: str, metadata: Any, save_images: bool = True) -> list[Any]:
    """Download images and save metadata to a JSON file."""
    print("Downloading and saving images...")

    # Create the output directory
    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    # Prepare data for JSON
    output_data = []

    # Display a row of the merged dataframe
    print(merged_df.head())

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing rows"):
        try:
            # Get image
            image = row["image"]  # Assuming image is in PIL.Image.Image format
            if not isinstance(image, Image.Image):
                raise ValueError(f"Invalid image object for ID {row['id']}.")

            # Save image to disk
            if save_images:
                image_path = os.path.join(output_dir, f"{row['id']}.jpg")
                image.save(image_path)
        except Exception as e:
            print(f"Error processing ID {row['id']}: {e}")

    # Get the output data from metadata
    metadata_df = pd.DataFrame.from_dict(metadata, orient="index").reset_index()
    metadata_df.rename(columns={"index": "id"}, inplace=True)
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing metadata"):
        try:
            output_data.append(
                {
                    "id": row["id"],
                    "image_path": os.path.join(output_dir, f"{row['id']}.jpg"),
                    "image_description": row["image_description"],
                    "attributes": row["attributes"],
                }
            )
        except Exception as e:
            print(f"Error processing ID {row['id']}: {e}")

    return output_data


def save_to_json(output_data: list[Any], output_json_path: str) -> None:
    """Save the processed data to a JSON file."""
    print(f"Saving processed data to {output_json_path}...")
    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)
    print(f"Processed data saved to {output_json_path}.")


if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Process and save images and metadata.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vector-institute/newsmediabias-plus-clean",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--metadata_json_path", type=str, required=True, help="Path to the metadata JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed images")
    parser.add_argument("--output_json_path", type=str, required=True, help="Output path for processed metadata JSON")
    args = parser.parse_args()
    # Configurations
    # Hugginf Face dataset name
    dataset_name = args.dataset_name
    # Path to the metadata JSON file
    data_folder = os.path.dirname(args.metadata_json_path)
    metadata_json_path = os.path.join(data_folder, args.metadata_json_path)
    # Output directory for processed images
    output_dir = os.path.join(data_folder, args.output_dir)
    # Output path for processed metadata JSON
    output_json_path = os.path.join(data_folder, args.output_json_path)

    # Load datasets
    dataset_df = load_huggingface_dataset(dataset_name)
    metadata = load_json_file(metadata_json_path)

    # Merge metadata with images
    merged_df = merge_with_images(metadata, dataset_df)

    # Download images and collect metadata - set `save_images` to False to skip saving images
    output_data = download_and_save_images(merged_df, output_dir, metadata, save_images=False)

    # Save processed data to JSON
    save_to_json(output_data, output_json_path)

# Note: The script assumes that the images are in PIL format and that the metadata JSON file is structured correctly.
# Make sure to adjust the paths and dataset names as needed.

# To run the script, use the following command:
# python eval2_QA_generation_preprocessing.py \
#     --dataset_name <dataset_name> \
#     --metadata_json_path <metadata_json_path> \
#     --output_dir <output_dir> \
#     --output_json_path <output_json_path>

"""HumanI Bench — task 6 (empathetic captioning): resize images."""

import argparse
import os
from collections.abc import Sequence

from PIL import Image


def resize_images(input_folder: str, output_folder: str, target_size: Sequence[int] = (350, 350)) -> None:
    """Resize images."""
    os.makedirs(output_folder, exist_ok=True)

    for count, file_name in enumerate(os.listdir(input_folder)):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            input_path = os.path.join(input_folder, file_name)
            try:
                with Image.open(input_path) as img:
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    new_name = f"image_{count + 1}.jpg"
                    output_path = os.path.join(output_folder, new_name)
                    img_resized.save(output_path)
                    print(f"Processed {file_name} -> {new_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images to a fixed size and rename them.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing original images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save resized images")
    args = parser.parse_args()

    resize_images(args.input_folder, args.output_folder)

# Example usage:
# python resize_images.py \
#     --input_folder ./empathy_dataset/images \
#     --output_folder ./empathy_dataset/resized


# To run this script, save it as resize_images.py and execute the following command in your terminal:
# python resize_images.py \
#     --input_folder <path_to_your_input_folder> \
#     --output_folder <path_to_your_output_folder>

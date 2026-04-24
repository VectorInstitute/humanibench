"""
Apply image perturbations with imgaug (blur, noise, motion blur, compression, blackout, etc.).

Input images are read from a folder; outputs are saved with the perturbation name in each filename.
"""

import argparse
import os

import imageio
import imgaug.augmenters as iaa
from PIL import Image


# Define imgaug augmenters
AUGMENTERS = {
    "blur": iaa.GaussianBlur(sigma=(0.0, 2.5)),
    "noise": iaa.AdditiveGaussianNoise(scale=0.1 * 255),
    "motion_blur": iaa.MotionBlur(k=10),
    # "scale_0.5": iaa.Resize(0.5),
    # "scale_2": iaa.Resize(2.0),
    "compression": iaa.JpegCompression(compression=90),
    "blackout": iaa.CoarseSaltAndPepper(0.2, size_percent=(0.1, 0.1)),
}


def apply_augmentation(input_folder: str, output_folder: str, perturbation: str) -> None:
    """Apply augmentation."""
    os.makedirs(output_folder, exist_ok=True)

    perturbations_to_apply = list(AUGMENTERS.keys()) if perturbation == "all" else [perturbation]

    for pert in perturbations_to_apply:
        augmenter = AUGMENTERS[pert]
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for filename in image_files:
            input_path = os.path.join(input_folder, filename)

            # Read image
            image = imageio.imread(input_path)

            # Apply augmentation
            augmented_image = augmenter(image=image)

            # Create new filename with augmentation label
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{pert}{ext}"
            output_path = os.path.join(output_folder, new_filename)

            # Save result
            Image.fromarray(augmented_image).save(output_path)

        print(f"'{pert}' to {len(image_files)} image(s).")


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply image perturbations using imgaug.")
    parser.add_argument("--input_folder", required=True, help="Folder with input images.")
    parser.add_argument("--output_folder", required=True, help="Folder to save augmented images.")
    parser.add_argument(
        "--perturbation",
        default="all",
        choices=["blur", "noise", "motion_blur", "compression", "blackout", "all"],
        help="Type of perturbation to apply (default: all).",
    )
    args = parser.parse_args()

    apply_augmentation(args.input_folder, args.output_folder, args.perturbation)

# To run the script:
# python image_perturbation.py --input_folder path/to/input --output_folder path/to/output --perturbation blur

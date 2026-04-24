"""HumanI Bench — task 5 (visual grounding): annotate with yolo."""

import argparse
import glob
import json
import os
import random
import uuid
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from matplotlib import patches
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


def extract_images(
    dataset_name: str,
    output_dir: str,
    num_images: int = 1000,
    filename_map_path: str | None = None,
) -> int:
    """Extract images."""
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset(dataset_name)
    records = ds["train"].shuffle(seed=42).select(range(num_images))

    print(f"Extracting {num_images} images...")
    filename_map = {}
    extracted = 0

    for i, record in enumerate(tqdm(records)):
        image_data = record.get("image", None)
        if not image_data:
            continue

        filename = f"{uuid.uuid4().hex[:10]}.jpg"
        image_path = os.path.join(output_dir, filename)

        try:
            if isinstance(image_data, dict) and "bytes" in image_data:
                img = Image.open(BytesIO(image_data["bytes"]))
            elif hasattr(image_data, "convert"):
                img = image_data
            elif isinstance(image_data, np.ndarray):
                img = Image.fromarray(image_data)
            else:
                print(f"Skipping record {i} (invalid format: {type(image_data)})")
                continue

            img.save(image_path)
            filename_map[filename] = {
                "record_id": i,
                "title": record.get("title", ""),
                "url": record.get("image_url", ""),
            }
            extracted += 1

        except Exception as e:
            print(f"Error saving image for record {i}: {e}")

    if filename_map_path:
        with open(filename_map_path, "w") as f:
            json.dump(filename_map, f, indent=4)

    print(f"✅ Extracted {extracted} images to {output_dir}")
    return extracted


def run_yolo_inference(
    image_dir: str, output_json: str, model_path: str = "yolov8n.pt", conf_threshold: float = 0.6
) -> list[dict[str, Any]]:
    """Run yolo inference."""
    model = YOLO(model_path)
    image_paths = sorted(
        [f for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for f in glob.glob(os.path.join(image_dir, ext))]
    )

    print(f"Found {len(image_paths)} images to process")
    results = []

    for path in tqdm(image_paths):
        try:
            result = model(path)
            img_width, img_height = Image.open(path).size
            boxes = result[0].boxes.cpu().numpy()

            annotations = []
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                annotations.append(
                    {"class_name": class_name, "bbox": [float(x1), float(y1), float(x2), float(y2)], "confidence": conf}
                )

            results.append(
                {
                    "id": os.path.basename(path),
                    "image_path": f"/{os.path.basename(path)}",
                    "original_filename": os.path.basename(path),
                    "img_width": img_width,
                    "img_height": img_height,
                    "annotations": annotations,
                }
            )

        except Exception as e:
            print(f"❌ Failed to process {path}: {e}")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    total = len(results)
    total_detections = sum(len(r["annotations"]) for r in results)
    with_detections = sum(1 for r in results if r["annotations"])

    print(f"\n✅ Saved results to {output_json}")
    print(f"Total images processed: {total}")
    print(f"Images with detections ≥ {conf_threshold}: {with_detections} ({(with_detections / total) * 100:.1f}%)")
    print(f"Total detections: {total_detections}")
    return results


def visualize_results(results: list[dict[str, Any]], image_dir: str, num_to_visualize: int = 5) -> None:
    """Visualize results."""
    print(f"Visualizing {num_to_visualize} detections...")
    sample_results = random.sample(results, min(num_to_visualize, len(results)))

    for entry in sample_results:
        img_path = os.path.join(image_dir, entry["id"])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = plt.imread(img_path)
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)

        for ann in entry["annotations"]:
            x1, y1, x2, y2 = ann["bbox"]
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            plt.text(
                x1,
                y1 - 5,
                f"{ann['class_name']} ({ann['confidence']:.2f})",
                color="white",
                fontsize=10,
                bbox={"facecolor": "red", "alpha": 0.5},
            )

        plt.title(f"Detections: {entry['id']}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def parse_args() -> Any:
    """Parse args."""
    parser = argparse.ArgumentParser(description="Run YOLO inference on NMB+ dataset.")
    parser.add_argument("--image_dir", type=str, default="images", help="Directory to store extracted images")
    parser.add_argument("--output_json", type=str, default="yolo_detections.json", help="Output JSON path")
    parser.add_argument("--num_images", type=int, default=2500, help="Number of images to extract")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLOv8 model path or name")
    parser.add_argument("--confidence", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--visualize", action="store_true", help="Visualize a few detection results")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional Hugging Face token")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    filename_map_path = os.path.join(args.image_dir, "filename_mapping.json")
    extract_images("vector-institute/newsmediabias-plus-clean", args.image_dir, args.num_images, filename_map_path)
    results = run_yolo_inference(args.image_dir, args.output_json, args.yolo_model, args.confidence)

    if args.visualize:
        visualize_results(results, args.image_dir)

"""HumanI Bench — task 5 (visual grounding): calc metrics."""

import argparse
import json

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


# ------------------------------
# Calculate Intersection over Union (IoU)
# ------------------------------
def calculate_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calculate iou."""
    if box_a is None or box_b is None:
        return 0.0
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ------------------------------
# Compute Average Precision at a given IoU threshold
# ------------------------------
def compute_ap(iou_scores: list[float], iou_threshold: float) -> float:
    """Compute ap."""
    y_true = np.array([1 if iou >= iou_threshold else 0 for iou in iou_scores])
    if y_true.sum() == 0:
        return 0.0
    y_scores = np.ones_like(y_true, dtype=float)  # dummy scores
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate(results_path: str) -> None:
    """Evaluate."""
    with open(results_path, "r") as f:
        data = json.load(f)

    ious = []
    num_total = len(data)
    num_failed = 0
    correct_05 = 0
    correct_075 = 0

    for entry in data:
        gt = entry.get("ground_truth_bbox")
        pred = entry.get("predicted_bbox")

        if not (pred and len(pred) == 4):
            num_failed += 1
            continue

        iou = calculate_iou(gt, pred)
        ious.append(iou)
        if iou >= 0.5:
            correct_05 += 1
        if iou >= 0.75:
            correct_075 += 1

    num_valid = len(ious)
    miou = np.mean(ious) if ious else 0.0
    acc_05 = correct_05 / num_valid if num_valid else 0.0
    acc_075 = correct_075 / num_valid if num_valid else 0.0

    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = {t: compute_ap(ious, t) for t in thresholds}
    map_50 = aps[0.5]
    map_50_95 = np.mean(list(aps.values()))

    # Print Metrics
    print("Detection Evaluation")
    print("------------------------------------------------")
    print(f"Total Samples                : {num_total}")
    print(f"Valid Predictions            : {num_valid}")
    print(f"Failed / Missing Predictions : {num_failed} ({num_failed / num_total:.2%})")
    print(f"Mean IoU                     : {miou:.4f}")
    print(f"Accuracy @ IoU ≥ 0.50        : {acc_05:.2%}")
    print(f"Accuracy @ IoU ≥ 0.75        : {acc_075:.2%}\n")

    print("Average Precision (AP) by IoU threshold:")
    for t, ap in aps.items():
        print(f"  AP@{t:.2f} = {ap:.4f}")

    print(f"\nmAP@0.50                      : {map_50:.4f}")
    print(f"mAP@[.5:.95]                  : {map_50_95:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bounding box predictions.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results JSON file.")
    args = parser.parse_args()
    results_path = args.results_path
    evaluate(results_path)

# To run the script:
# python calc_metrics.py --results_path <path_to_results_json_file>

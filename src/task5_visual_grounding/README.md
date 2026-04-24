# Eval 3: Visual Grounding

This task evaluates how accurately a model links textual references to visual regions using bounding boxes.

# Project Structure
```
.
├── data_preparation
│   └── annotate-with-yolo.py
├── inferences
│   ├── aya_vl_detection.py
│   ├── gemini_detection.py
│   ├── janus_vl_detection.py
│   ├── llava_detection.py
│   ├── openai_detection.py
│   ├── phi4_detection.py
│   └── qwen2.5_vl_detection.py
└── metrics
    ├── calc_metrics.py
    └── score.py
```


# Key Components

## 1.  Image and Question Preparation (data_preparation/)
Scripts for annotating images with bounding boxes using YOLOv8. This step is crucial for preparing the dataset for evaluation.


## 2. Model Inference (inferences/)
Scripts to run evaluation across multiple VQA models (Qwen, LLaVA, DeepSeek, Phi-4, Gemini, GPT-4o), following a shared structure:

- Model loading from Hugging Face or local path
- Image preprocessing using PIL or base64
- Prompt formatting to generate answers and reasoning
- Save generated answers and reasoning.
- Handles missing or corrupted samples gracefully.

## 3. Metric Computation (metrics/calc_metrics.py)
Evaluates single bounding box predictions
Reports:
- Mean IoU (mIoU)
- Accuracy @ IoU ≥ 0.5 and 0.75
- Average Precision (AP) at multiple thresholds
- mAP@[.5:.95]

## 4. Metric Computation (metrics/score.py)
Evaluates multi-object detection (e.g. LLaVA/DeepSeek formats)
- Computes Precision, Recall, F1, and Avg IoU
- Supports per-class metrics
- Saves output JSON with overall and per-class stats



# How to Run

## 1. (Optional) Data Preparation
```bash
python annotate-with-yolo.py \
  --image_dir images \
  --output_json yolo_detections.json \
  --num_images 1000 \
  --yolo_model yolov8n.pt \
  --confidence 0.5 \
  --visualize \
  --hf_token <your_huggingface_token>

```

## 2. Run Inference on Models
Example for Aya Vision 8B:
```bash
python aya_vl_detection.py \
  --input_file <path_to_input_json> \
  --output_file <path_to_output_json> \
  --iou_threshold <iou_threshold> \
```

Similarly run for other models

Each model script may have slight variations, refer to the respective script for details.

## 3. Compute Statistics
```bash
python metrics/calc_metrics.py \
  --results_path path/to/your/results.json
```

```bash
python metrics/score.py \
  --input_file path/to/predictions.json \
  --output_file path/to/save/eval_results.json \
  --iou_threshold 0.2
```


# Requirements
- Python 3.8+
    - torch
    - transformers
    - tqdm
    - sklearn
    - pandas
    - Pillow (PIL)

> **Note:**
> Exact package versions are not fixed in this repository.
> Different models may require slightly different versions of libraries (especially `torch` and `transformers`).
>
> You can find the specific environment requirements for each model at their Hugging Face pages.
> For running specific models, please check their respective Hugging Face pages and install any additional requirements if needed.

# Outputs
- JSON files per model containing:
    - Predictions
- CSV summary containing:
-   Mean IoU
    - Accuracy @ IoU ≥ 0.5 and 0.75
    - Average Precision (AP) at multiple thresholds
    - mAP@[.5:.95]

# Questions?
If you have any questions or need further assistance, please open an issue in this repository or contact the maintainers.

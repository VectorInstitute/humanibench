# Eval 7: Image Resilience Evaluation

This task evaluates Visual Question Answering (VQA) model robustness against different types of image perturbations. It uses multiple open-source models (like Aya Vision 8B, Gemma 3 12B, Llama 3.2 Vision, Phi-4) and OpenAI's batch API for evaluation.

# Project Structure
```
.
├── data_preparation
│   └── image_perturbation.py         # Script to create perturbed versions of images
├── evaluation
│   ├── eval7_openAI.py                # Prepares evaluation batches and sends them to OpenAI's batch API
│   ├── postprocess.py                 # Postprocesses model outputs for uniform formatting
│   └── process_evaluation.py          # Summarizes evaluation metrics like average score, matches
└── inferences
    ├── aya_vision_8b.py               # Evaluation script for Aya Vision 8B
    ├── gemma3_12b.py                  # Evaluation script for Gemma 3 12B
    ├── Llama.py                       # Evaluation script for Llama 3.2-11B Vision
    └── Phi4.py                        # Evaluation script for Phi-4 Multimodal
```


# Key Components

## 1. Image Perturbation (data_preparation/image_perturbation.py)
Applies transformations like:
- Gaussian Blur
- Gaussian Noise
- Motion Blur
- JPEG Compression
- Blackout/Noise Salt

Generates augmented images from a clean dataset to simulate real-world degradations.


## 2. Model Inference (inferences/)
Scripts to run evaluation for each model individually. Common behavior:

- Load pre-trained or local model weights
- Resize and preprocess images
- Generate answers for a given question-image pair
- Save raw results (Predicted Answer, Ground Truth, Question)

## 3. OpenAI Evaluation (evaluation/eval7_openAI.py)
- Prepares structured prompts that compare a degraded answer vs a clean reference answer
- Encodes images into Base64
- Submits evaluation batches using OpenAI's Batch API (gpt-4o-mini)
- Receives structured evaluation outputs: match/mismatch, score (1-10), missing visual elements, explanation.

## 4. Post-processing Results (evaluation/postprocess.py)
- Cleans noisy outputs from different models (removes unwanted tags, splits reasoning from answer)
- Handles model-specific parsing rules

## 5. Final Evaluation Summarization (evaluation/process_evaluation.py)
- Calculates:
    - Total matches (Yes/No)
    - Average score (out of 10)
- Exports results to a CSV summary table.


# How to Run

## 1. Perturb Images
```bash
python data_preparation/image_perturbation.py \
  --input_folder <path_to_clean_images> \
  --output_folder <path_to_save_augmented_images> \
  --perturbation all
```

## 2. Run Inference on Models
Example for Aya Vision 8B:
```bash
python aya_vision_8b.py \
    --dataset <path_to_dataset> \
    --image_folder <path_to_image_folder> \
    --save_path <path_to_save_results> \
    --attack <attack_type>
```

Similarly run for:
- gemma3_12b.py
- Llama.py
- Phi4.py

(Same command structure, just the model script changes.)

## 3. Postprocess Model Outputs
```bash
python evaluation/postprocess.py
```

Cleans raw outputs for consistent evaluation.

## 4. Evaluate with OpenAI

```bash
python eval7_openAI.py \
    --data_folder <path_to_your_data_folder> \
    --openai_api_key <your_openai_api_key>
```

Generates batch jobs for OpenAI evaluation API.

## Summarize Evaluation Results
```bash
python process_evaluation.py \
    --directory_path <path_to_your_directory_containing_json_files>
```

Creates a final CSV summary table with match rates and average scores.


# Requirements
- Python 3.8+
    - torch
    - transformers
    - tqdm
    - imgaug
    - PIL (Pillow)
    - openai
    - pandas

> **Note:**
> Exact package versions are not fixed in this repository.
> Different models may require slightly different versions of libraries (especially `torch` and `transformers`).
>
> You can find the specific environment requirements for each model at their Hugging Face pages:
> - [Aya Vision 8B](https://huggingface.co/CohereForAI/aya-vision-8b)
> - [Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it)
> - [Llama-3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
> - [Phi-4 Multimodal Instruct](https://huggingface.co/microsoft/phi-4-multimodal-instruct)
>
> For running specific models, please check their respective Hugging Face pages and install any additional requirements if needed.


# Notes
- Some models expect local weights (e.g., Aya Vision 8B), so adjust MODEL_DIR if needed.
- Batch processing is optimized for OpenAI's API limits (~250 per batch).
- Evaluation scripts automatically handle missing or problematic samples.
- Intermediate results are saved every few samples to avoid data loss during long runs.

# Outputs
- Cleaned result JSON files per model
- CSV file summarizing:
    - Average match score
    - Number of perfect matches

# Questions?
If you have any questions or need further assistance, please open an issue in this repository or contact the maintainers.

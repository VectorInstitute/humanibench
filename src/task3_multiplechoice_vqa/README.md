# Eval 3: Multiple Choice Visual Question Answering

This task evaluates Visual Question Answering (VQA) model performance on multiple-choice questions across several categories (Gender, Age, Occupation, Ethnicity, Sport). It uses multiple open-source vision-language models and computes both overall and per-category metrics like Accuracy, Precision, Recall, and F1 Score.

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

## 1.  Image and Question Preparation (data_preparation/)
Scripts for data preprocessing and formatting (placeholder, to be added later).


## 2. Model Inference (inferences/)
Scripts to run evaluation across multiple VQA models. Each script shares a common structure:

- Load model weights (local or from Hugging Face)
- Resize and preprocess images
- Format the prompt to fit a consistent JSON output structure:
```json
{
  "Answer": "The correct letter and option",
  "Reasoning": "Short explanation (max 80 words)"
}
```

- Save generated answers and reasoning.
- Handles missing or corrupted samples gracefully.

## 3. Metric Computation (metrics/compute_stat_eval.py)
- Computes Accuracy, Precision, Recall, and F1 Score.
- Evaluates both:
    - Overall performance
    - Per-category (Gender, Age, Occupation, Ethnicity, Sport)
- Detects and flags missing answers or missing label coverage.
- Merges predicted and ground truth answers based on ID and Attribute.


## 4. Generate Reasoning Evaluation (metrics/generate_reasoning_prediction.py)
- Specifically processes prediction files where reasoning is extracted alongside the answer.
- Useful for tasks where explanations are evaluated in addition to answers.
> Run it similarly as compute_stat_eval.py.


# How to Run

## 1. (Optional) Data Preparation
(Placeholder: To be added in the future.)

## 2. Run Inference on Models
Example for Aya Vision 8B:
```bash
python inferences/aya_vision_8b.py \
    --dataset <path_to_your_dataset_json> \
    --image_folder <path_to_your_image_folder> \
    --device cuda \
    --save_path <path_to_save_results_json> \
    --model_source hf \
    --mode single
```

Similarly run for:
- CogVLM2_Llama3_Chat_19B.py
- Deepseek_VL2.py
- glm_4v_9B.py
- etc.

Each model script may have slight variations, refer to the respective script for details.

## 3. Compute Statistics
```bash
python metrics/compute_stat_eval.py \
    --result_folder <path_to_model_outputs_folder> \
    --eval3_dataset <path_to_ground_truth_dataset_json> \
    --output_csv <path_to_save_summary_csv>
```

```bash
python metrics/generate_reasoning_prediction.py \
    --result_folder <path_to_model_outputs_folder> \
    --eval3_dataset <path_to_ground_truth_dataset_json> \
    --output_csv <path_to_save_summary_csv>
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
> You can find the specific environment requirements for each model at their Hugging Face pages:
> - [Aya Vision 8B](https://huggingface.co/CohereForAI/aya-vision-8b)
> - [Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it)
> - [Llama-3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
> - [Phi-4 Multimodal Instruct](https://huggingface.co/microsoft/phi-4-multimodal-instruct)
>
> For running specific models, please check their respective Hugging Face pages and install any additional requirements if needed.


# Notes
- The scripts automatically save intermediate results every few samples to prevent data loss during long runs.
- Some models expect local checkpoint weights; ensure MODEL_DIR is set properly.
- Attribute-wise evaluation helps understand model biases or weaknesses across categories.
- Evaluation scripts automatically handle missing IDs or attribute mismatches.

# Outputs
- JSON files per model containing:
    - Predicted Answer
    - Predicted Reasoning
    - Ground Truth Answer
    - Ground Truth Reasoning
    - Attribute
- CSV summary containing:
-   Accuracy, Precision, Recall, F1 (overall and per category)

# Questions?
If you have any questions or need further assistance, please open an issue in this repository or contact the maintainers.

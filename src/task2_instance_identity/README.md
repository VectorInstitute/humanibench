# Eval 2: Instance Identity

This task focuses on evaluating the context understanding capabilities of various models. The evaluation is based on a dataset of images and associated questions, where the models are tasked with generating answers based on the visual content and social context of the images.

# Project Structure
```
.
├── data_preparation
│   ├── eval2_QA_generation_preprocessing.py     # Prepares dataset and images from metadata
│   └── eval2_QA_generation_postprocessing.py    # Postprocesses generated QA pairs
├── inferences
│   ├── aya_vision_8b.py                         # Inference script for Aya Vision 8B
│   ├── gemma3_12b.py                            # Inference script for Gemma 3 12B
│   ├── CogVLM2_Llama3_Chat_19B.py                # Inference script for CogVLM2-Llama3
│   ├── InternVL2_5_8B.py                        # Inference script for InternVL2.5 8B
│   ├── Deepseek_VL2.py                          # Inference script for Deepseek VL-2
│   ├── Qwen2_5_v2.py                            # Inference script for Qwen 2.5 v2
│   ├── Llava_v1_6_7B.py, Llava_v1_6_13B.py       # Inference scripts for Llava 7B and 13B
│   ├── and others...                            # (e.g., Magma 8B, Phi-4, JanusPro, etc.)
│   ├── postprocessing
│   │   ├── convert_to_csv.py                    # Converts model outputs to CSV
│   │   └── postprocess.py                       # Cleans and formats outputs
└── metrics
    ├── deepeval_scores.py                        # Measures Bias, Toxicity, Relevance, Faithfulness
    └── stat_scores.py                            # Measures FrugalScore, BERTScore, METEOR
```


# Key Components

## 1. Data Preparation (data_preparation/)

- <i>eval2_QA_generation_preprocessing.py</i>:
Loads metadata and dataset, processes images, and prepares structured JSON for evaluation.
- <i>eval2_QA_generation_postprocessing.py</i>:
Postprocesses model-generated questions, extracting clean question-answer pairs.


## 2. Model Inference (inferences/)

Scripts for running evaluation across different models:

- Load model weights (locally or from Hugging Face).
- Resize and preprocess images.
- Generate answers for a given image and question.
- Save outputs including:
    - Predicted Answer
    - Ground Truth Answer
    - Question
    - Model Reasoning

Supported models include Aya Vision 8B, Deepseek VL2, Gemma 3 12B, Phi-4, Llava, CogVLM2, InternVL2.5, Qwen2.5, Magma 8B, Paligemma, etc.

## 3. Postprocessing (inferences/postprocessing/)

- convert_to_csv.py and postprocess.py:
Convert raw JSON results into standardized CSV format for evaluation.

## 4. Metrics Evaluation (metrics/)

- <i>deepeval_scores.py</i><br>
    Evaluates model answers using:
    - Bias Metric
    - Toxicity Metric
    - Answer Relevance
    - Faithfulness
- <i>stat_scores.py</i><br>
    Evaluates answers using standard NLP metrics:
    - FrugalScore
    - BERTScore
    - METEOR

Both scripts process large datasets in batches and resume if interrupted.

# How to Run

## 1. Data Preparation
Preprocess and download metadata + images:
```bash
python data_preparation/eval2_QA_generation_preprocessing.py \
  --dataset_name <huggingface_dataset_name> \
  --metadata_json_path <path_to_metadata.json> \
  --output_dir <output_folder_for_images> \
  --output_json_path <output_metadata.json>
```

Postprocess generated QA pairs:
```bash
python data_preparation/eval2_QA_generation_postprocessing.py \
  --input_csv <path_to_generated_questions_csv> \
  --metadata_json <path_to_metadata_json> \
  --output_json <path_to_save_postprocessed_json>
```

## 2. Model Inference
Example command for running Aya Vision 8B:
```bash
python inferences/aya_vision_8b.py \
  --dataset <path_to_dataset_json> \
  --image_folder <path_to_images> \
  --device cuda \
  --save_path <path_to_save_results.json> \
  --model_source local
```

Similarly run for:
- gemma3_12b.py
- Llama.py
- Phi4.py

(Same command structure, just the model script changes.)

## 3. Postprocess Model Outputs

Convert generated model outputs to CSV:

```bash
python inferences/postprocessing/convert_to_csv.py \
  <input_folder_with_json_files> \
  <output_folder_for_csv_files>
```

Cleans raw outputs for consistent evaluation.

## 4. Evaluate with Deepeval Metrics

```bash
python metrics/deepeval_scores.py \
  --input <path_to_input_csv> \
  --output <path_to_save_evaluation_csv>
```

## 5. Evaluate with Statistical Metrics
```bash
python metrics/stat_scores.py \
  --input <path_to_input_csv> \
  --output <path_to_save_statistical_metrics_json> \
  --batch_size <batch_size>
```

# Requirements
- Python 3.8+
    - torch
    - transformers
    - tqdm
    - pandas
    - PIL (Pillow)
    - openai
    - deepeval
    - evaluate
    - datasets

> **Note:**
> Model-specific dependencies (especially for torch and transformers versions) can vary.
Check the Hugging Face page of each model for detailed requirements:
>
> You can find the specific environment requirements for each model at their Hugging Face pages:
> - [Aya Vision 8B](https://huggingface.co/CohereForAI/aya-vision-8b)
> - [Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it)
> - [Llama-3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
> - [Phi-4 Multimodal Instruct](https://huggingface.co/microsoft/phi-4-multimodal-instruct)
> - ... so on for other models.
>
> For running specific models, please check their respective Hugging Face pages and install any additional requirements if needed.


# Notes
- Set save_images=False in preprocessing if you do not want to save images locally.
- Intermediate outputs are saved during inference to avoid data loss.
- Deepeval-based evaluation can automatically resume if interrupted.
- OpenAI API key is required for deepeval scoring (stored in .env).

# Outputs
- Processed JSON datasets
- Model prediction CSVs
- Deepeval metrics CSVs (bias, toxicity, relevance, faithfulness)
- Statistical metrics JSONs (FrugalScore, BERTScore, METEOR)

# Questions?
If you have any questions or need further assistance, please open an issue in this repository or contact the maintainers.

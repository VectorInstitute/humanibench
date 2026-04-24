# Eval 4: Language and Culture - Multilingual Visual QA Evaluation

This task evaluates Visual Question Answering (VQA) models in a multilingual setting, covering both simple QA and multiple-choice QA formats.
It supports models like Aya Vision 8B, Gemma 3 12B, Llama 3.2 Vision, and Phi-4, with evaluation based on translation quality, reasoning, and multilingual consistency.
It supports both open-ended and multiple-choice questions, with a focus on generating and evaluating multilingual datasets (Eval2 and Eval3).

# Project Structure
```
.
├── data_preparation
│   ├── eval5_eval2QAgen.py            # Batch generation of translated QA pairs (Eval2 - simple QA)
│   ├── eval5_eval3QAgen.py            # Batch generation of translated MCQ pairs (Eval3 - multiple-choice QA)
│   ├── postprocess_eval2.py           # Postprocess OpenAI results for Eval2
│   └── postprocess_eval3.py           # Postprocess OpenAI results for Eval3
├── inferences
│   ├── aya_vision_8b_eval2.py          # Inference script for Eval2 QA using Aya Vision 8B
│   ├── aya_vision_8b_eval3.py          # Inference script for Eval3 MCQ using Aya Vision 8B
│   ├── gemma3_12b_eval2.py             # Inference script for Eval2 QA using Gemma 3 12B
│   ├── gemma3_12b_eval3.py             # Inference script for Eval3 MCQ using Gemma 3 12B
│   ├── Llama_eval2.py                  # Inference script for Eval2 QA using Llama 3.2
│   ├── Llama_eval3.py                  # Inference script for Eval3 MCQ using Llama 3.2
│   ├── Phi4_eval2.py                   # Inference script for Eval2 QA using Phi-4 Multimodal
│   └── Phi4_eval3.py                   # Inference script for Eval3 MCQ using Phi-4 Multimodal
└── metrics
    ├── compute_stat_eval.py            # Compute evaluation metrics (accuracy, F1, etc.)
    ├── convert_to_lang_eval2.py         # Language cleaning & conversion for Eval2 results
    ├── convert_to_lang_eval3.py         # Language cleaning & conversion for Eval3 results
    ├── get_stats_deepeval.py            # DeepEval scoring (bias, faithfulness, answer relevancy)
    └── run_deepeval_Eval2.py            # Run DeepEval analysis for Eval2 outputs
```


# Key Components

## 1. Multilingual QA Generation (data_preparation/)
- <i>eval5_eval2QAgen.py:</i>
Generates simple QA pairs in the target language from English using OpenAI's batch API.
- <i>eval5_eval3QAgen.py:</i>
Generates multiple-choice questions, options, answers, and reasoning in the target language.

Both scripts handle batch preparation, prompting, and submission via OpenAI API.


## 2.  Postprocessing of OpenAI Results (data_preparation/)
- <i>postprocess_eval2.py:</i>
Cleans and extracts translated question-answer pairs for Eval2.
- <i>postprocess_eval3.py:</i>
Cleans MCQ outputs (Question, Options, Answer, Reasoning) for Eval3.

Handles noisy or incomplete outputs, ensuring consistent formatting.

## 3. Model Inference (inferences/)
Scripts for running model inference on Eval2 (simple QA) and Eval3 (MCQ) separately. Common steps:

- Load the pre-trained models (locally or from Hugging Face).
- Preprocess input (resize images, construct multilingual prompts).
- Generate predictions.
- Save predictions along with metadata (ID, attribute, etc.).


## 4. Metrics Calculation (metrics/)
- <i>compute_stat_eval.py:</i>
Calculates accuracy, precision, recall, F1 score overall and per attribute (Gender, Age, Ethnicity, Occupation, Sport).
- <i>convert_to_lang_eval2.py / convert_to_lang_eval3.py:</i>
Further clean outputs, remove English text contamination, and prepare final evaluation files.

## 5. DeepEval Evaluation (metrics/)
- <i>run_deepeval_Eval2.py:</i>
Runs bias, answer relevancy, and faithfulness checks using DeepEval framework.
- <i>get_stats_deepeval.py:</i>
Summarizes DeepEval scores overall and per attribute, generating CSV reports.



# How to Run

## 1. Generate Multilingual QA/MCQ
```bash
python data_preparation/eval5_eval2QAgen.py \
  --data_path <path_to_english_data> \
  --language <target_language> \
  --openai_api_key <your_openai_api_key> \
  --image_folder <path_to_image_folder>
```

or for MCQ:
```bash
python data_preparation/eval5_eval3QAgen.py \
  --data_path <path_to_english_mcq_data> \
  --language <target_language> \
  --openai_api_key <your_openai_api_key> \
  --images_folder <path_to_image_folder>
```
This will generate the translated QA/MCQ pairs in the specified target language.

## 2. Postprocess Translated Outputs
```bash
python data_preparation/postprocess_eval2.py \
  --jsonl_dir <path_to_jsonl_files> \
  --output_dir <path_to_output> \
  --input_folder <folder_with_language_specific_json> \
  --base_file <path_to_english_base_data>
```

(Similar for postprocess_eval3.py.)

## 3. Run Model Inference
Example for Aya Vision 8B:
```bash
python inferences/aya_vision_8b_eval2.py \
  --dataset <path_to_translated_eval2_dataset> \
  --image_folder <path_to_images> \
  --save_path <path_to_save_predictions> \
  --device cuda \
  --model_source local
```

and similarly for Eval3.

(Same command structure, just the model script changes.)

## 3. Postprocess Model Outputs
```bash
python metrics/convert_to_lang_eval2.py \
  --input_folder_path <path_to_model_outputs> \
  --output_folder_path <path_to_cleaned_outputs>
```

or for Eval3:
```bash
python metrics/convert_to_lang_eval3.py \
  --input_folder_path <path_to_model_outputs> \
  --output_folder_path <path_to_cleaned_outputs>
```

## 4. Compute Standard Metrics for Eval3

```bash
python metrics/compute_stat_eval.py \
  --result_folder <path_to_results_folder> \
  --eval3_dataset_folder <path_to_eval3_dataset_folder> \
  --output_csv <output_csv_path>
```

## 5. DeepEval Analysis for Eval2
```bash
python metrics/run_deepeval_Eval2.py \
  --input <path_to_eval2_translations> \
  --output <path_to_output_csv> \
  --english_reference <path_to_english_reference>
```
Summarise:
```bash
python metrics/get_stats_deepeval.py
```

This will generate a CSV file with the DeepEval scores for each model.

## Summarize Evaluation Results
```bash
python process_evaluation.py \
    --directory_path <path_to_your_directory_containing_json_files>
```

This will create a final CSV summary table with match rates and average scores for each model.

# Requirements
- Python 3.8+
    - torch
    - transformers
    - tqdm
    - PIL (Pillow)
    - openai
    - pandas
    - openai
    - deepeval
    - scikit-learn
    - pandas
    - langdetect

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
- OpenAI batch API is used heavily for translation.
- Intermediate results are saved every few steps to prevent data loss.
- Language detection and cleaning scripts ensure high-quality translations.
- DeepEval metrics (Bias, Faithfulness, Relevancy) add an extra quality dimension.

# Outputs
- Translated QA and MCQ datasets.
- Model predictions (JSON files).
- CSV files summarizing:
    - Accuracy, Precision, Recall, F1 scores (overall and per attribute).
    - DeepEval Bias, Faithfulness, Answer Relevancy scores.

# Questions?
If you have any questions or need further assistance, please open an issue in this repository or contact the maintainers.

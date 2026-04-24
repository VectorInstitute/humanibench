# Eval 1: Scene Understanding and Social Bias Evaluation

This task evaluates visual models on their ability to understand scenes and infer social categories (such as gender, age, ethnicity, occupation, and sport participation) with rich, nuanced reasoning. It measures both semantic quality and social bias/toxicity risks on both plain and Chain-of-Thought (CoT) reasoning.

# Project Structure
```
.
├── data_preparation
│   ├── generate_attributes.py                # Generate visible social attributes from images
│   ├── generate_captions_descriptions.py      # Generate image captions and descriptions
│   └── gpt.py                                 # Generate reasoning-based answers using GPT models
├── inferences
│   ├── aya_vision_inference.py                # Inference script for Aya Vision 8B
│   ├── cogvlm2_inference.py                   # Inference script for CogVLM-2
│   ├── deepseek_vl2_small_inference.py        # Inference script for DeepSeek-VL-2 Small
│   ├── glm4v_inference.py                     # Inference script for GLM-4V
│   ├── instructblip_inference.py              # Inference script for InstructBLIP
│   ├── InternVL2.5_inference.py               # Inference script for InternVL-2.5
│   ├── janus_inference.py                     # Inference script for Janus
│   ├── llama3.2_11B_inference.py              # Inference script for Llama-3.2 11B Vision
│   ├── llama3.2_90B_inference.py              # Inference script for Llama-3.2 90B Vision
│   ├── llava_inference.py                     # Inference script for LLaVA models
│   ├── Molmo_7B_inference.py                  # Inference script for Molmo-7B
│   ├── openai_small_inference.py              # Inference script for OpenAI Vision small models
│   ├── paligemma_inference.py                 # Inference script for PaliGemma
│   ├── phi3.5_VL_inference.py                 # Inference script for Phi-3.5 VL
│   ├── phi4_inference.py                      # Inference script for Phi-4
│   └── qwen2.5_vl_inference.py                # Inference script for Qwen-2.5 VL
├── metrics
│   ├── accuracy.py                            # Script to compute BERTScore (semantic similarity)
│   └── run_evalSOE.py                         # Script to evaluate bias and toxicity
└── README.md
```


# Key Components

## 1. Data Preparation (data_preparation/)
Scripts to generate inputs for model evaluations:

- <i>generate_attributes.py</i>:
Uses GPT-4o to predict which social attributes (Gender, Age, Ethnicity, Occupation, Sport) are visible in the image.
- <i>generate_captions_descriptions.py</i>:
Generates a one-line caption and a full-paragraph description for each image using a local vision-language model.
- <i>gpt.py</i>:
Uses GPT-4o to answer a set of reasoning questions about each image, guided by the detected attributes.



## 2. Model Inference (inferences/)

Scripts to run evaluation for multiple VLMs (vision-language models).

Each script does:

- Load pre-trained model and processor.
- Take image and corresponding questions as input.
- Produce multi-turn reasoning answers (for both plain and CoT prompts).
- Save results (Predicted Answer, Question, Attribute Category) incrementally.
> Note:
> All inference scripts follow the same overall logic as aya_vision_inference.py, with minor model-specific adaptations.

## 3. Metrics and Evaluation (metrics/)
- <i>accuracy.py</i>:
Calculates BERTScore between model-generated answers and GPT ground-truth answers to measure semantic similarity.
- <i>run_evalSOE.py</i>:
Evaluates bias and toxicity risks in generated model outputs using the DeepEval library and OpenAI API.



# How to Run

## 1. Generate Attributes
```bash
python data_preparation/generate_attributes.py \
  --dataset_name <huggingface_dataset_name> \
  --split <split> \
  --batch_size 32 \
  --openai_api_key <your_openai_api_key> \
  --save_path <path_to_save_attributes.json> \
  --attributes_file <path_to_existing_attributes.json>
```

## 2. Generate Captions and Descriptions
```bash
python data_preparation/generate_captions_descriptions.py \
  --dataset_name <huggingface_dataset_name> \
  --split <split> \
  --batch_size 32 \
  --save_path <path_to_save_captions.json> \
  --hf_token <your_huggingface_token> \
  --model_path <model_path_to_caption_model>

```


## 3. Generate GPT Reasoning Answers
```bash
python data_preparation/gpt.py \
  --hf_dataset_name <huggingface_dataset_name> \
  --hf_cache_dir <path_to_hf_cache> \
  --selected_samples_path <path_to_selected_samples.json> \
  --results_path <path_to_save_results.json>
```

## 4. Run Model Inference
Example for Aya Vision 8B:
```bash
python inferences/aya_vision_inference.py \
  --hf_token <your_huggingface_token> \
  --model_path CohereForAI/aya-vision-8b \
  --results_file <path_to_save_results.json> \
  --dataset_name <huggingface_dataset_name> \
  --selected_samples <path_to_selected_samples.json>
```

Similarly run for:
- cogvlm2_inference.py
- deepseek_vl2_small_inference.py
- glm4v_inference.py
- etc.

## 5. Calculate Semantic Similarity (BERTScore)
```bash
python metrics/accuracy.py \
  --input <path_to_predicted_answers.csv> \
  --output <path_to_save_output.csv> \
  --ground_truth <path_to_ground_truth.csv>
```

## 6. Evaluate Bias and Toxicity
```bash
python metrics/run_evalSOE.py \
  --input <path_to_model_responses.csv> \
  --output <path_to_save_bias_toxicity.csv>
```


# Requirements
- Python 3.8+
    - torch
    - transformers
    - huggingface_hub
    - datasets
    - openai
    - deepeval
    - bert_score
    - tqdm
    - dotenv
    - pandas
    - PIL (Pillow)

>Important:
>Different models may require slightly different versions of libraries (especially torch, transformers, etc.).
>Please check each model’s Hugging Face page for specific environment needs.


# Notes
- Attribute and caption generation uses GPT-4o and local captioning models respectively.
- Model inference is modular across VLMs, using a consistent template for question-answering.
- Evaluation metrics (semantic + social bias) are separate and modular.
- Scripts automatically save intermediate outputs to minimize data loss during long runs.

# Outputs
- Attributes JSON
- Captions/Descriptions JSON
- Reasoning answers JSON
- Inference outputs JSON
- Semantic evaluation CSV (BERTScore)
- Bias and toxicity evaluation CSV

# Questions?
If you have any questions or need further assistance, please open an issue in this repository or contact the maintainers.

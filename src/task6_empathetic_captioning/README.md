# Eval 6: Empathetic Captioning Evaluation

This task evaluates model performance in generating both factual and empathetic captions for empathy-related images.
It uses multiple open-source models (like Aya Vision 8B, DeepSeek-Vision, Llama 3.2 Vision, Phi-4) and an external linguistic tool (LIWC) for emotion analysis.

# Project Structure
```
.
├── data_preparation
│   ├── get_images.py              # Download empathy-related news images using GDELT API
│   ├── resize_images.py            # Resize images to standard dimensions
│   └── generate_captions.py        # Generate simple captions via OpenAI GPT models
├── inferences
│   ├── aya_vision.py               # Inference script for Aya Vision 8B
│   ├── deepseek.py                 # Inference script for DeepSeek-Vision
│   ├── llama.py                    # Inference script for Llama 3.2 Vision
│   └── phi4.py                     # Inference script for Phi-4 Multimodal
├── metrics
│   └── postprocess
│       └── to-csv.py               # Converts model outputs into a unified CSV
└── README.md
```


# Key Components

## 1. Data Preparation (data_preparation/)
- <i>get_images.py</i>:
Downloads empathy-related news articles and images using GDELT API based on predefined empathy categories.
- <i>resize_images.py</i>:
Resizes all collected images to a standard 350×350 pixel size for uniform model input.
- <i>generate_captions.py</i>:
Uses OpenAI's gpt-4o-mini model to generate simple factual captions for each image (to serve as a baseline).



## 2. Model Inference (inferences/)
Scripts to generate both factual and empathetic captions from each model:

- Load pretrained models from Hugging Face (Aya Vision 8B, DeepSeek, Llama 3.2, Phi-4)
- For each image:
    - Generate two styles of captions:
    - Factual Caption: Objective description
    - Empathetic Caption: Human-centered, emotional description

Save outputs in structured JSON format (id, image path, model captions, CSV baseline captions)

## 3. Post-processing (metrics/postprocess/to-csv.py)
- Converts the model-generated JSON files into a CSV table format.
- Extracts key fields:
    - Image ID
    - Image Path
    - Baseline Captions (simple, emphatic)
    - Model Captions (simple, empathetic)


## 4. Metrics and Evaluation
- LIWC (Linguistic Inquiry and Word Count) tool is used externally to evaluate emotional richness in captions.
- No internal metric calculation scripts are included.
- CSV outputs from this project can be uploaded directly into LIWC for emotional and linguistic feature analysis.


# How to Run

## 1. Download Empathy Images
```bash
python data_preparation/get_images.py \
    --output_csv <path_to_save_empathy_news_csv>
```

## 2. Resize Images
Example for Aya Vision 8B:
```bash
python data_preparation/resize_images.py \
    --input_folder <path_to_original_images> \
    --output_folder <path_to_resized_images>
```

## 3. Generate Baseline Captions using OpenAI
```bash
python data_preparation/generate_captions.py \
    --image_dir <path_to_resized_images> \
    --save_path <path_to_save_captions_csv>
```

## 4. Run Inference on Models
Example for Aya Vision 8B:
```bash
python inferences/aya_vision.py \
    --hf_token <your_huggingface_token> \
    --model_path CohereForAI/aya-vision-8b \
    --csv_file <path_to_captions_csv> \
    --results_file <path_to_save_results_json> \
    --image_folder <path_to_resized_images>
```

Similarly run for:
- deepseek.py
- llama.py
- phi4.py
- (Just change the script name and model path accordingly.)


## 5. Postprocess Model Outputs
```bash
python metrics/postprocess/to-csv.py \
    --json_filename <path_to_model_output_json> \
    --csv_filename <path_to_output_csv>
```

## 6. Evaluate Emotion (External)
- Upload the final CSV into LIWC Tool for emotional, psychological, and linguistic analysis.
- Compare simple vs empathetic captions for emotional richness scores.

# Requirements
- Python 3.8+
    - torch
    - transformers
    - pandas
    - newspaper3k
    - PIL (Pillow)
    - openai
    - tqdm
    - requests
    - dotenv

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
- Some scripts (like generate_captions.py) use OpenAI's API — ensure .env file contains your API key.
- Image downloads rely on the GDELT API — replace USER_AGENT and GDELT_API constants in get_images.py.
- Hugging Face token is required to run local models.
- Intermediate results are automatically saved to prevent data loss during long runs.

# Outputs
- JSON file per model containing:
    - Image ID
    - Baseline captions
    - Model-generated captions
- Final CSV file summarizing all captions
- LIWC-compatible CSV ready for emotional feature analysis

# Questions?
If you have any questions or need further assistance, please open an issue in this repository or contact the maintainers.

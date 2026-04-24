"""HumanI Bench — task 3 (multiple-choice VQA): InternVL2 5 8B."""

import json
import logging
import math
import os
import re
import time
from argparse import ArgumentParser
from typing import Any

import torch
import torchvision.transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed


set_seed(45)
MAX_NEW_TOKENS = 150

# Model directory for local and Hugging Face
MODEL_DIR = "model-weights/InternVL2_5-8B"
HF_MODEL_ID = "OpenGVLab/InternVL2_5-8B"

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT = """Answer the question using one of the given choices based on the image.

Question:
{QUESTION}

Provide the response only in the following JSON format:
{{"Answer": "The correct letter and option",
"Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

Do not provide any other extra information.
"""


def extract_answer_and_reason(text: str) -> tuple[str, str | None]:
    """Extract the answer and reason from the VLM response."""
    try:
        # Find the first '{' and the last '}' to extract a JSON string.
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            answer = data.get("Answer", "").strip()
            reasoning = data.get("Reasoning", "").strip()
            if answer and reasoning:
                return answer, reasoning

    except Exception:
        pass

    # If not JSON then extract the text directly using regex
    pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'

    match = re.search(pattern, text)

    if match:
        answer = match.group(1)
        reasoning = match.group(2)
        return answer, reasoning
    return text, None


def split_model(model_name: str) -> dict[str, int]:
    """Split model."""
    device_map: dict[str, int] = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


# Load the model and processor
def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load model."""
    print(f"Loading InternVL2.5-8B model from {'local directory' if model_source == 'local' else 'Hugging Face'}...")

    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID

    device_map = split_model(model_path.split("/")[-1])

    # Load model with necessary configurations
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
            cache_dir=os.environ["HF_HOME"],
        )
        .eval()
        .cuda()
    )

    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    return model, tokenizer


def build_transform(input_size: int) -> Any:
    """Build transform."""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            torchvision.transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    """Find closest aspect ratio."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Any, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False
) -> list[Any]:
    """Dynamic preprocess."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios_set = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    sorted_ratios: list[tuple[int, int]] = sorted(target_ratios_set, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, sorted_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> Any:
    """Load image."""
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)


def process_sample(
    model: Any, tokenizer: Any, img_path: str, question: str, input_size: int = 448, max_num: int = 12
) -> str:
    """Process sample."""
    try:
        # Load and process the image
        image = Image.open(img_path).convert("RGB")
        transform = build_transform(input_size=input_size)

        # Use dynamic preprocessing as in the previous code
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values).cuda()

        # Cast to BFloat16 to match model's expected tensor type
        pixel_values = pixel_values.to(torch.bfloat16)

        question_prompt = PROMPT.format(QUESTION=question)

        generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}

        # single-image single-round conversation (单图单轮对话)
        message = f"<image>\n{question_prompt}"
        return model.chat(tokenizer, pixel_values, message, generation_config)

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


# Main function to process dataset
def evaluate(
    model: Any, tokenizer: Any, dataset: list[Any], image_folder: str, save_path: str, mode: str = "single"
) -> None:
    """Evaluate."""
    results = []
    logger.info(f"Starting evaluation in {mode} mode...")
    intermediate_results_path = save_path.replace(".json", "_intermediate.json")
    prev_path = ""
    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")

            if not os.path.exists(img_path):
                logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
                continue

            question_with_choices = data["Question"] + "\n" + data["Options"]

            response = process_sample(model, tokenizer, img_path, question_with_choices)
            answer, reason = extract_answer_and_reason(response)

            results.append(
                {
                    "ID": data["ID"],
                    "Question": question_with_choices,
                    "Predicted_Answer": answer,
                    "Predicted_Reasoning": reason,
                    "Ground_Truth_Answer": data["Answer"],
                    "Ground_Truth_Reasoning": data["Reasoning"],
                    "Attribute": data["Attribute"],
                }
            )

            # Save intermediate results every 50
            if i % 10 == 0:
                intermediate_results_path = save_path.replace(
                    ".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(intermediate_results_path, "w") as f:
                    json.dump(results, f, indent=4, default=str)
                if prev_path != "":
                    os.remove(prev_path)
                logger.info(f"Intermediate results saved to {intermediate_results_path} and deleted {prev_path}.")
                prev_path = intermediate_results_path

            pbar.update(1)

    # Save results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    logger.info(f"Results saved to {save_path}.")


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./data/eval3/QA_Eval3.json", help="Path to dataset")
    parser.add_argument("--image_folder", type=str, default="./data/processed_images", help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--save_path", type=str, default="./results/results_InternVL2_5_8B.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="hf", help="Model source: 'local' or 'hf'")
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "batch"], help="Single or batch processing"
    )

    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Model source
    model_source = "huggingface" if args.model_source == "hf" else "local"
    logger.info(f"Model source: {model_source}")

    # Load model
    model, tokenizer = load_model(model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Conditional Execution: Single or Batch Processing
    logger.info("Running Single-Image Processing Mode...")

    evaluate(model, tokenizer, dataset, args.image_folder, args.save_path, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

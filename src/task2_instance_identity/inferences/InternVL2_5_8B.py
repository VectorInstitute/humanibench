"""HumanI Bench — task 2 (instance identity): InternVL2 5 8B."""

import json
import logging
import math
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
import torchvision.transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# Model paths for local and Hugging Face
MODEL_DIR = "/projects/NMB-Plus/E-VQA/model-weights/InternVL2_5-8B"
HF_MODEL_ID = "OpenGVLab/InternVL2_5-8B"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
offload_folder = ""


def split_model(model_name: str) -> dict[str, int]:
    """Generate a device map for model parallelism."""
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
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    # Map common modules to GPU 0
    for key in [
        "vision_model",
        "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.output",
        "language_model.model.norm",
        "language_model.model.rotary_emb",
        "language_model.lm_head",
    ]:
        device_map[key] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load and return the model and tokenizer."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading InternVL2_5-8B model from {model_path}...")
    device_map = split_model(model_path.split("/")[-1])
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def build_transform(input_size: int) -> Any:
    """Build a transformation pipeline for image preprocessing."""
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
    """Find the target aspect ratio closest to the image's aspect ratio."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
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
    """Split the image into blocks based on aspect ratio and target size."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios_set = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    sorted_ratios: list[tuple[int, int]] = sorted(target_ratios_set, key=lambda x: x[0] * x[1])
    target_aspect = find_closest_aspect_ratio(aspect_ratio, sorted_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect[0]
    target_height = image_size * target_aspect[1]
    blocks = target_aspect[0] * target_aspect[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> Any:
    """Load an image and return a tensor of pixel values."""
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def process_sample(
    model: Any, tokenizer: Any, img_path: str, question: str, input_size: int = 448, max_num: int = 12
) -> str:
    """Generate an answer for a given image and question."""
    try:
        image = Image.open(img_path).convert("RGB")
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values).cuda().to(torch.bfloat16)
        user_prompt = (
            f"Given question, answer in the following format:"
            f"Question:{question}"
            f"Answer:<answer> Reasoning:<reasoning> in the context of the image."
        )
        generation_config = {"max_new_tokens": 256, "do_sample": False}
        message = f"<image>\n{user_prompt}"
        response = model.chat(tokenizer, pixel_values, message, generation_config)
        print(f"User: {question}\nAssistant: {response}")
        return response
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(model: Any, tokenizer: Any, dataset: list[Any], image_folder: str, save_path: str) -> None:
    """Process the dataset and save evaluation results to a JSON file."""
    results = []
    logger.info("Starting evaluation...")
    prev_path = ""
    for i, data in enumerate(tqdm(dataset, unit="sample")):
        img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
            continue
        answer = process_sample(model, tokenizer, img_path, data["Question"])
        results.append(
            {
                "ID": data["ID"],
                "Question": data["Question"],
                "Predicted_Answer": answer,
                "Ground_Truth": data["Answer"],
                "Attribute": data["Attribute"],
            }
        )
        if i % 10 == 0:
            intermediate_results_path = save_path.replace(
                ".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(intermediate_results_path, "w") as f:
                json.dump(results, f, indent=4, default=str)
            if prev_path:
                os.remove(prev_path)
            logger.info(f"Intermediate results saved to {intermediate_results_path} and deleted {prev_path}.")
            prev_path = intermediate_results_path
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Results saved to {save_path}.")


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./data/eval2/QA_Eval2.json", help="Path to dataset")
    parser.add_argument("--image_folder", type=str, default="./data/processed_images", help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--save_path", type=str, default="./results/results_InternVL2_5_8B.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model_source)
    model.to(device)
    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    dataset = dataset[:10]  # For testing, use a small subset
    logger.info(f"Loaded dataset with {len(dataset)} samples.")
    evaluate(model, tokenizer, dataset, args.image_folder, args.save_path)
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

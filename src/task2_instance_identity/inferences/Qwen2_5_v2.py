"""HumanI Bench — task 2 (instance identity): Qwen2 5 v2."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# Model paths
MODEL_DIR = "/model-weights/Qwen2.5-VL-7B-Instruct/"
HF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Hugging Face cache directories
os.environ["HF_HOME"] = "/scratch/ssd004/scratch/aravindn/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/ssd004/scratch/aravindn/huggingface_cache"


def load_model(model_source: str = "local") -> tuple[Any, Any]:
    """Load and return the Qwen2.5 VL model and processor."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading Qwen2.5 model from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, cache_dir=os.environ["TRANSFORMERS_CACHE"], device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", revision="refs/pr/24", cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    return model, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize an image while preserving its aspect ratio."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, processor: Any, img_path: str, question: str) -> str:
    """Generate an answer for a single image-question pair."""
    try:
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"
        user_prompt = (
            f"Given question, answer in the following format:"
            f"Question:{question}"
            f"Answer:<answer> Reasoning:<reasoning> in the context of the image."
        )
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[prompt], images=image_inputs, padding=True, return_tensors="pt").to("cuda")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(
    model: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str, mode: str = "single"
) -> None:
    """Process the dataset and save evaluation results to a JSON file."""
    results = []
    logger.info(f"Starting evaluation in {mode} mode...")
    prev_path = ""
    for i, data in enumerate(tqdm(dataset, unit="sample")):
        img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
            continue
        answer = process_sample(model, processor, img_path, data["Question"])
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
            intermediate_path = save_path.replace(".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(intermediate_path, "w") as f:
                json.dump(results, f, indent=4, default=str)
            if prev_path:
                os.remove(prev_path)
            logger.info(f"Intermediate results saved to {intermediate_path} and deleted {prev_path}.")
            prev_path = intermediate_path
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Results saved to {save_path}.")


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="/projects/NMB-Plus/E-VQA/data/eval2/QA_Eval2.json", help="Path to dataset"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/projects/NMB-Plus/E-VQA/data/processed_images",
        help="Path to image folder",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--save_path", type=str, default="results/results_Qwen2_5.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "batch"], help="Single or batch processing"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, processor = load_model(args.model_source)
    model.to(device)

    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    logger.info(f"Loaded dataset with {len(dataset)} samples.")
    evaluate(model, processor, dataset, args.image_folder, args.save_path, args.mode)
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

"""HumanI Bench — task 2 (instance identity): Deepseek VL2."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from PIL import Image
from tqdm import tqdm


# Model paths
MODEL_DIR = "/model-weights/Qwen2.5-VL-7B-Instruct/"  # Local model path
HF_MODEL_ID = "deepseek-ai/deepseek-vl2-small"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""


def load_model(model_source: str = "local") -> tuple[Any, Any, Any]:
    """Load and return the model, processor, and tokenizer."""
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading Deepseek model from {model_path}...")
    processor = DeepseekVLV2Processor.from_pretrained(model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    tokenizer = processor.tokenizer
    model = (
        DeepseekVLV2ForCausalLM.from_pretrained(
            model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"], trust_remote_code=True
        )
        .to(torch.bfloat16)
        .cuda()
        .eval()
    )
    return model, processor, tokenizer


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize an image while preserving its aspect ratio."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, tokenizer: Any, processor: Any, img_path: str, question: str) -> str:
    """Process a single image-question pair and generate an answer."""
    try:
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"

        user_prompt = (
            f"Given question, answer in the following format:"
            f"Question:{question}"
            f"Answer:<answer> Reasoning:<reasoning> in the context of the image."
        )
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{user_prompt}", "images": [image]},
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepared_inputs = processor(conversations=conversation, images=[image], force_batchify=True).to(model.device)

        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**prepared_inputs)
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepared_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
            )
        return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def evaluate(model: Any, tokenizer: Any, processor: Any, dataset: list[Any], image_folder: str, save_path: str) -> None:
    """Evaluate the model on the dataset and save results to a JSON file."""
    results = []
    logger.info("Starting evaluation...")
    prev_path = ""
    for i, data in enumerate(tqdm(dataset, unit="sample")):
        img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
            continue
        answer = process_sample(model, tokenizer, processor, img_path, data["Question"])
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
        "--save_path", type=str, default="./results/results_Deepseek_VL2.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, processor, tokenizer = load_model(args.model_source)
    model.to(device)

    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    logger.info(f"Loaded dataset with {len(dataset)} samples.")
    evaluate(model, tokenizer, processor, dataset, args.image_folder, args.save_path)
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

"""HumanI Bench — task 2 (instance identity): JanusPro."""

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

import torch
from janus.models import VLChatProcessor
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM


# Model paths
MODEL_DIR = "/model-weights/Janus-Pro-7B/"  # Yet to download the model
HF_MODEL_ID = "deepseek-ai/Janus-Pro-7B"  # Hugging Face Model ID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
offload_folder = ""


def load_model(model_source: str = "local") -> tuple[Any, Any, Any]:
    """Load and return the model, tokenizer, and processor."""
    logger.info(f"Loading Janus-Pro model from {'local directory' if model_source == 'local' else 'Hugging Face'}...")
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    processor = VLChatProcessor.from_pretrained(model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    tokenizer = processor.tokenizer
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"], trust_remote_code=True
        )
        .to(torch.bfloat16)
        .cuda()
        .eval()
    )
    return model, tokenizer, processor


def resize_image(img_path: str, max_size: tuple[int, int] = (350, 350)) -> Any:
    """Resize image while preserving aspect ratio."""
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None


def process_sample(model: Any, tokenizer: Any, processor: Any, img_path: str, question: str, device: Any) -> str:
    """Generate answer for one image-question pair."""
    try:
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"
        pil_images = [image]
        user_prompt = (
            f"Given question, answer in the following format:"
            f"Question:{question}"
            f"Answer:<answer> Reasoning:<reasoning> in the context of the image."
        )
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{user_prompt}", "images": pil_images},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepared_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)
        inputs_embeds = model.prepare_inputs_embeds(**prepared_inputs)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer if answer else "No answer generated"
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"


def process_batch(
    model: Any, tokenizer: Any, processor: Any, dataset: list[Any], image_folder: str, device: Any
) -> list[Any]:
    """Process a batch of data and return generated answers."""
    batch_conversations = []
    image_list = []
    for data in dataset:
        img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
            continue
        image = resize_image(img_path)
        if image is None:
            continue
        user_prompt = (
            f"Given question, answer in the following format:"
            f"Question:{data['Question']}"
            f"Answer:<answer> Reasoning:<reasoning> in the context of the image."
        )
        batch_conversations.append(
            {"role": "<|User|>", "content": f"<image_placeholder>\n{user_prompt}", "images": [image]}
        )
        image_list.append(image)
    if not batch_conversations:
        return []
    prepared_inputs = processor(conversations=batch_conversations, images=image_list, force_batchify=True).to(
        model.device
    )
    inputs_embeds = model.prepare_inputs_embeds(**prepared_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepared_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=150,
        do_sample=False,
        use_cache=True,
    )
    return tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)


def evaluate(
    model: Any, tokenizer: Any, processor: Any, dataset: list[Any], image_folder: str, device: Any, save_path: str
) -> None:
    """Process dataset and save results as JSON."""
    results = []
    logger.info("Starting evaluation...")
    prev_path = ""
    for i, data in enumerate(tqdm(dataset, unit="sample")):
        img_path = os.path.join(image_folder, f"{data['ID']}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for ID {data['ID']} at {img_path}")
            continue
        answer = process_sample(model, tokenizer, processor, img_path, data["Question"], device)
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
            intermediate_path = save_path.replace(".json", f"_intermediate_{i}_{time.strftime('%Y%m%d%H%M%S')}.json")
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
    parser.add_argument("--dataset", type=str, default="./data/eval2/QA_Eval2.json", help="Path to dataset")
    parser.add_argument("--image_folder", type=str, default="./data/processed_images", help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--save_path", type=str, default="./results/results_Janus_Pro.json", help="Output file to save results"
    )
    parser.add_argument("--model_source", type=str, default="hf", help="Model source: 'local' or 'hf'")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer, processor = load_model(args.model_source)
    model.to(device)

    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    dataset = dataset[:10]
    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    evaluate(model, tokenizer, processor, dataset, args.image_folder, device, args.save_path)
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

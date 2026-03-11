import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import json
import gc
from tqdm import tqdm
import logging 

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from qwen_omni_utils import process_mm_info

TEST_PROMPT_DAILYOMNI = """
Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.

Question: {question}
Choices:
{options_str}

Your answer should be a capital letter representing your choice: A, B, C, or D.
Don't generate any other text.
"""

# =========================
# Hyperparameters
# =========================

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
TOTAL_PIXELS = 32 * 768 * 28 * 28
NFRAMES = 32
MAX_NEW_TOKENS = 64


# =========================
# Dataset
# =========================

class DailyOmniDataset(Dataset):
    def __init__(self, json_path, video_root):
        self.video_root = video_root
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        qid = idx
        video_id = item["video_id"]
        question = item["Question"]
        options = item["Choice"]
        gt_letter = item["Answer"]

        options_str = "\n".join(options)

        prompt = TEST_PROMPT_DAILYOMNI.format(
            question=question,
            options_str=options_str
        )

        video_path = os.path.join(self.video_root,video_id,f"{video_id}_video.mp4")
        audio_path = os.path.join(self.video_root, video_id, f"{video_id}_audio.wav")

        message = [
            dict(type="text", value=prompt),
            dict(type="audio", value=audio_path),
            dict(type="video", value=video_path),
        ]

        return message, idx, {
            "qid": qid,
            "video_id": video_id,
            "question": question,
            "options": options,
            "gt_letter": gt_letter,
            "video_duration": item['video_duration']
        }


def collate_fn(batch):
    messages, indices, metas = zip(*batch)
    return list(messages), list(indices), list(metas)


# =========================
# Model Loader
# =========================

def _load_model(model_path,device):
    if "qwen2.5_omni" in model_path.lower():
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None, 
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    elif "qwen3_omni" in model_path.lower():
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map=None,  
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    else:
        raise ValueError("Unsupported model")

    model.eval()
    return model, processor


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--daily_json", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./dailyomni_eval/results")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    # 配置日志：每个 GPU 一个独立的 log 文件
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gpu_{args.gpu_id}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8')]
    )
    logger = logging.getLogger(__name__)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    dataset = DailyOmniDataset(json_path=args.daily_json, video_root=args.video_root)
    subset_indices = [i for i in range(len(dataset)) if i % args.num_gpus == args.gpu_id]
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

    dataloader = DataLoader(
        subset_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True     
    )

    logger.info(f"GPU {args.gpu_id} started. Handling {len(subset_dataset)} items.")
    model, processor = _load_model(args.model_path, device)
    actual_model = model.module if hasattr(model, "module") else model

    os.makedirs(args.output_dir, exist_ok=True)

    pbar = tqdm(
        total=len(dataloader), 
        desc=f"GPU {args.gpu_id}", 
        position=args.gpu_id, 
        leave=True
    )

    for messages_batch, indices, metas_batch in dataloader:
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):
            qid = meta["qid"]
            video_id = meta['video_id']
            
            save_name = f"{video_id}_task{qid}.json"
            save_path = os.path.join(args.output_dir, save_name)
            
            if os.path.exists(save_path):
                logger.info(f"Skipping qid={qid} | video={video_id} (Processed)")
                pbar.update(1)
                continue

            logger.info(f"Processing qid={qid} | video={video_id}")

            content = []
            for item in message:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["value"]})
                elif item["type"] == "audio":  
                    content.append({"type": "audio", "audio": item["value"]})
                elif item["type"] == "video":
                    content.append({
                        "type": "video",
                        "video": item["value"],
                        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                        "total_pixels": TOTAL_PIXELS, "max_frames": NFRAMES,
                    })

            new_message = [{"role": "user", "content": content}]
            text = processor.apply_chat_template([new_message], tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(new_message, use_audio_in_video=False)

            inputs = processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=False, use_audio_in_video=False
            )

            inputs = {k: v.to(device).contiguous() if torch.is_tensor(v) else v for k, v in inputs.items()}
            model_dtype = actual_model.dtype
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=model_dtype)

            try:
                with torch.no_grad():
                    generated_ids = actual_model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True, use_audio_in_video=False
                    )
                    if isinstance(generated_ids, tuple): generated_ids = generated_ids[0]
                    input_len = inputs["input_ids"].shape[1]
                    new_token_ids = generated_ids[0, input_len:]
                    response = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

                result = {
                    "qid": qid, "video_id": video_id, "question": meta["question"],
                    "options": meta["options"], "gt_letter": meta["gt_letter"],
                    "prediction": response, "raw_response": response
                }
                with open(save_path, "w", encoding="utf-8") as fw:
                    json.dump(result, fw, ensure_ascii=False, indent=2)

                logger.info(f"qid {qid} | pred: {response} | gt: {meta['gt_letter']}")

            except Exception as e:
                logger.error(f"Error processing qid {qid}: {e}")

            pbar.update(1)
            del inputs
            if 'generated_ids' in locals(): del generated_ids
            torch.cuda.empty_cache()
            gc.collect()

    pbar.close() 
    logger.info(f"GPU {args.gpu_id} all finished.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import json
import gc
from tqdm import tqdm

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor
)
from qwen_omni_utils import process_mm_info

TEST_PROMPT_DAILYOMNI = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with **only the letter** (A, B, C, ...) of your choice.

Question: {question}
Options:
{options_str}
Answer:
"""
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
MAX_NEW_TOKENS = 512


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
            dict(type="video", value=video_path),
            dict(type="text", value=prompt),
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

def _load_model(model_path):
    if "qwen2.5_omni" in model_path.lower():
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    # elif "qwen3_omni" in model_path.lower():
    #     model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    #         model_path,
    #         dtype="auto",
    #         device_map="auto",
    #         attn_implementation="flash_attention_2",
    #     )
    #     processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    # else:
    #     raise ValueError("Unsupported model")

    model.eval()
    return model, processor


def collate_fn(batch):
    messages, indices, metas = zip(*batch)
    return list(messages), list(indices), list(metas)

def get_lr(audio_id, video_id, input_ids):
    input_ids = torch.as_tensor(input_ids)  
    v_lst = (input_ids == video_id).nonzero(as_tuple=True)[0]
    a_lst = (input_ids == audio_id).nonzero(as_tuple=True)[0]
    v_lst = v_lst.tolist()
    a_lst = a_lst.tolist()
    return v_lst, a_lst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument('--DailyOmni_json', type=str, required=True, help="DailyOmni JSON path")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nframes', type=int, default=32)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./results")

    args = parser.parse_args()
    dataset = DailyOmniDataset(
        json_path=args.DailyOmni_json,
        video_root=args.video_root
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Loading model: {args.model_path}")
    model, processor = _load_model(args.model_path)
    actual_model = model.module if hasattr(model, "module") else model
    NFRAMES = args.nframes
    TOTAL_PIXELS =  NFRAMES * 768 * 28 * 28

    os.makedirs(args.output_dir, exist_ok=True)

    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(tqdm(dataloader, desc="Greedy Inference", total=len(dataloader))):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):

            qid = meta["qid"]
            print(f"Processing DailyOmni qid = {qid} | video = {meta['video_id']}")
            
            content = []
            for item in message:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["value"]})
                elif item["type"] == "audio":  
                    content.append({
                        "type": "audio",
                        "audio": item["value"],
                    })
                elif item["type"] == "video":
                    content.append({
                        "type": "video",
                        "video": item["value"],
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                        "total_pixels": TOTAL_PIXELS,
                        "max_frames": NFRAMES,
                    })

            new_message = []
            new_message.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            })
            new_message.append({'role': 'user', 'content': content})

            text = processor.apply_chat_template(
                new_message,
                tokenize=False,
                add_generation_prompt=True
            )

            audios, images, videos = process_mm_info(
                new_message,
                use_audio_in_video=True
            )

            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=False,
                use_audio_in_video=True
            )

            inputs = inputs.to(model.device).to(model.dtype)
            actual_model.eval()

            with torch.no_grad():
                generated_ids = model.generate(
                        **inputs, 
                        thinker_max_new_tokens = 2, 
                        use_audio_in_video=True, 
                        return_audio=False, 
                        temperature=1,
                    )
                
            prompt_length = inputs["input_ids"].shape[1]
            response = processor.tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)
            result = {
                "qid": meta["qid"],
                "video_id": meta["video_id"],
                "question": meta["question"],
                "options": meta["options"],
                "gt_letter": meta["gt_letter"],
                "prediction": response,
                "raw_response": response
            }

            save_name=f"{meta['video_id']}_task{qid}.json"
            save_path = os.path.join(args.output_dir, save_name)
            with open(save_path, "w", encoding="utf-8") as fw:
                json.dump(result, fw, ensure_ascii=False, indent=2)

            print(f"qid {qid} | pred: {response} | gt: {meta['gt_letter']}")

            del inputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()

    print("DailyOmni inference finished:", args.output_dir)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
import torch
import json
import os
import argparse
import gc
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
import decord
from PIL import Image
import random
import base64
from io import BytesIO
from collections import defaultdict 


TEST_PROMPT_DAILYOMNI = """
Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.

Question: {question}
Choices:
{options_str}

Your answer should be a capital letter representing your choice: A, B, C, or D.
Don't generate any other text.
"""

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
TOTAL_PIXELS = 32 * 768 * 28 * 28
NFRAMES = 128
MAX_NEW_TOKENS = 512

high_frames = 4
mid_frames = 8
low_frames = 32
sample_frames = 8

import sys
sys.path.append("/mnt/data2/yangmrl/project/video2text/Long-CLIP")
from model import longclip
device = torch.device("cuda:0")
model_path = "/mnt/data2/yangmrl/project/video2text/models/clip/longclip-B.pt"
clip_model, clip_processor = longclip.load(model_path, device=device)

class DailyOmniDataset(Dataset):
    def __init__(self, json_path, video_root):
        self.video_root = video_root
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            self.data = self.data

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

        # ⭐ 正确视频路径
        video_path = os.path.join(
            self.video_root,
            video_id,
            f"{video_id}_video.mp4"
        )

        message = [
            dict(type="text", value=prompt),
            dict(type="video", value=video_path),
            dict(type='audio', value=video_path), 
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

def _load_model(model_path):
    if "qwen2.5_omni" in model_path.lower():
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    elif "qwen3_omni" in model_path.lower():
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    else:
        raise ValueError("Unsupported model")

    model.eval()
    return model, processor

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames, frame_idx, frame_time, video_time

def TextImageMatching(text, images, tau=1.0):
    question = text
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        text = longclip.tokenize([question]).to(device)
        images = torch.stack([clip_processor(Image.fromarray(image)) for image in images]).to(device)
        
        image_features = clip_model.encode_image(images)
        text_features = clip_model.encode_text(text)
        logits_per_text = text_features @ image_features.T  # this is the image-text similarity score

    probs = (logits_per_text / tau).softmax(dim=1)[0]
    probs = torch.log(probs) - torch.log(-torch.log(torch.rand(len(images), device=probs.device) + 1e-10) + 1e-10)  # gumble
    indices = np.argsort(-probs.cpu().detach().numpy())

    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--daily_json", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./dailyomni_eval/results")
    parser.add_argument("--mode", type=str, default="all", choices=["video"])

    args = parser.parse_args()

    dataset = DailyOmniDataset(
        json_path=args.daily_json,
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
    actual_model = model  

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0")
    model_dtype = actual_model.dtype
    cnt = 0
    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(dataloader):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):
            cnt+=1
            # if cnt<=2209:
            #     continue
            video_id = meta["video_id"]
            qid = meta["qid"]
            # if video_id != 'BQDKdEep':
            #     continue
            print(f"Processing Dailyomni {video_id} ...")
            duration = meta['video_duration']
            seconds = 0
            for t in duration:
                if t == 's':
                    continue
                seconds = seconds*10+int(t)
            
            content = []
            for item in message:
                
                if item['type'] == 'text':
                    content.append({'type': 'text', 'text': item['value']})
                elif item['type'] == 'audio':
                    content.append({
                        'type': 'audio', 'audio': item['value'],
                        'audio_start':0,
                        'audio_end': seconds
                    })

            
            video_pth = f'/mnt/data2/yangmrl/project/video2text/test_data/dailyomni/Videos/{video_id}/'
            visual, frame_idx, frame_time, video_time = load_video(video_pth+video_id+"_video.mp4", NFRAMES)
            try:
                indices = TextImageMatching(meta['question'], visual, tau=0.8)
                visual_tmp = [None] * len(visual)
                visual = [Image.fromarray(v).convert("RGB") for v in visual]
                width, height = visual[0].size
                for idx in indices[:high_frames]:
                    visual_tmp[idx] = visual[idx].resize((width // 2, height // 2), Image.Resampling.LANCZOS)
                for idx in indices[high_frames: high_frames+mid_frames]:
                    visual_tmp[idx] =visual[idx].resize((width // 4, height // 4), Image.Resampling.LANCZOS)
                for idx in indices[high_frames+mid_frames: high_frames+mid_frames+low_frames]:
                    visual_tmp[idx] =visual[idx].resize((width // 8, height // 8), Image.Resampling.LANCZOS)
                visual = [v for v in visual_tmp if v is not None ]
            except Exception as e:
                if len(visual) >= sample_frames:
                    visual = visual[sorted(random.sample(range(len(visual)), sample_frames))]
                height, width, _ = visual[0].shape
                visual = [Image.fromarray(v).convert("RGB").resize((width // 2, height // 2), Image.Resampling.LANCZOS) for v in visual]
            # visual = [v for v in visual_tmp if v is not None ]
            image_content = []
            for base64_image in visual:
                # base64_image = Image.fromarray(v).convert("RGB")
                buffer = BytesIO()
                base64_image.save(buffer, format="JPEG")
                base64_bytes = base64.b64encode(buffer.getvalue())
                base64_string = base64_bytes.decode("utf-8")
                content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
            
            new_message = [{'role': 'user', 'content': content}]
            text = processor.apply_chat_template([new_message], tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(new_message, use_audio_in_video=(args.mode == "all"))
        
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=False,
                use_audio_in_video=(args.mode == "all")
            )
         
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=model_dtype)

            actual_model.eval()

            with torch.no_grad():
                past_key_values = None
                past_key_values_audio = None
                generated_ids = inputs['input_ids'].clone().to(device)
                new_token_list = []

                for step in range(MAX_NEW_TOKENS):
                    if step == 0:
                        outputs = actual_model.thinker(
                            **inputs,
                            output_attentions=True,  
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=True,
                            return_dict=True,
                            use_audio_in_video=(args.mode == "all"),
                        )
                    else:
                        current_inputs = {
                            'input_ids': inputs['input_ids'].to(device),
                            'attention_mask': inputs['attention_mask'].to(device),
                        }
 
                        outputs = actual_model.thinker(
                            **current_inputs,
                            output_attentions=True,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=False,
                            return_dict=True,
                            use_audio_in_video=(args.mode == "all"),
                        )
                    
                    hidden_states = outputs.hidden_states   
                    attention_weight = outputs.attentions
                    attn_lstlayer = attention_weight[-1]
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = next_token_logits.argmax(dim=-1, keepdim=True).to(device)
                
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=1).to(device)
                    past_key_values = outputs.past_key_values
                    new_token_list.append(next_token_id.item())

                    attention_mask = torch.cat(
                        [inputs['attention_mask'], torch.ones((inputs['attention_mask'].shape[0], 1), dtype=inputs['attention_mask'].dtype, device=device)],
                        dim=-1
                    )
                    inputs['attention_mask'] = attention_mask
                    inputs['input_ids'] = next_token_id
                    

                    if next_token_id.item() == processor.tokenizer.eos_token_id:
                        break

            response = processor.tokenizer.decode(new_token_list, skip_special_tokens=True).strip()

            result = {
                "qid": meta["qid"],
                "video_id": meta["video_id"],
                "question": meta["question"], 
                "options": meta["options"],
                "gt_letter": meta["gt_letter"],
                "prediction": response,
                "raw_response": response
            }

            save_path = os.path.join(args.output_dir, f"{video_id}_{qid}.json")
            with open(save_path, "w", encoding="utf-8") as fw:
                json.dump(result, fw, ensure_ascii=False, indent=2)

            print(f"{video_id} | pred: {response} | gt: {meta['gt_letter']}")

            del inputs, generated_ids, outputs
            torch.cuda.empty_cache()
            gc.collect()

    print("Dailyomni inference completed:", args.output_dir)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
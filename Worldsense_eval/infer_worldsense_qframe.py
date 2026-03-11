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


TEST_PROMPT_WORLDSENSE = """
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
device = torch.device("cuda:2")
model_path = "/mnt/data2/yangmrl/project/video2text/models/clip/longclip-B.pt"
clip_model, clip_processor = longclip.load(model_path, device=device)

class WorldSenseDataset(Dataset):
    def __init__(self, json_path: str, video_root: str = "/mnt/data2/yangmrl/project/video2text/test_data/worldsense/videos"):
        self.video_root = video_root
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)  # {video_id: {...}}

        self.samples = []
        for video_id, item in self.data.items():
            for task_name, task in item.items():
                if task_name.startswith("task"):
                    self.samples.append((video_id, task_name, task, item))  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx,mode = None):
        video_id, task_name, task, full_item = self.samples[idx]

        question = task["question"]
        candidates = task["candidates"]
        gt_answer = task["answer"]

        alphas = [chr(65 + i) + ". " for i in range(len(candidates))]
        options_str = "\n".join([a + c for a, c in zip(alphas, candidates)])

        prompt = TEST_PROMPT_WORLDSENSE.format(
            question=question,
            options_str=options_str
        )

        video_path = os.path.join(self.video_root, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        message = [
            dict(type='text', value=prompt),
            dict(type='video', value=video_path),
            dict(type='audio', value=video_path),  
        ]

        meta = {
            "video_id": video_id,
            "task_name": task_name,
            "question": question,
            "candidates": candidates,
            "gt_answer": gt_answer,
            "video_duration": full_item.get("video_duration", "unknown"),
            "domain": full_item.get("domain", "unknown"),
            "sub_category": full_item.get("sub_category", "unknown"),
        }

        return message, idx, meta


def _load_model(model_path, local_rank=None):
    device = torch.device(f"cuda:{local_rank}" if local_rank is not None else "cuda")
    if 'qwen2.5_omni' in model_path:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            attn_implementation='flash_attention_2'   
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model.eval()
    elif 'qwen3_omni' in model_path:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype = 'auto',
            device_map='auto',
            attn_implementation='flash_attention_2'   
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        model.eval()

    return model, processor


def collate_fn(batch):
    messages, indices, metas = zip(*batch)
    return list(messages), list(indices), list(metas)


def attn_analyze(v_lst,a_lst,attn):
    num_heads, seq_len, _ = attn.shape
    attn = attn.squeeze(0)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attn.device))

    attn_causal = attn * causal_mask  # shape (num_heads, seq_len, seq_len)
    combined = torch.mean(attn_causal, dim = 0)  # shape (seq_len, seq_len)
    combined_ = torch.mean(combined,dim = 0)
    
    top_pct = 0.1
    topk = max(int(seq_len * top_pct), 1)  

    cnt = 0;cnt_ = 0
    values, indices = torch.topk(combined_, topk, dim=-1)
    for i in range(len(indices)):
        if indices[i] in v_lst:
            cnt+=1
        elif indices[i] in a_lst:
            cnt_+=1
    
    print(cnt,cnt_)
    unique_top_ids = torch.unique(indices.flatten()).cpu().tolist()
    print("Top token IDs:", unique_top_ids)
    print(values)
    
    data = combined.cpu().numpy()

    mask = np.triu(np.ones_like(data, dtype=bool), k=1)
    plt.figure(figsize=(10, 10))
    sns.heatmap(data,mask = mask,cmap="viridis", square=True,
                xticklabels=False, yticklabels=False, vmin=1e-4, vmax=0.2)
    plt.title("Combined Attention (all heads)")
    plt.savefig("/mnt/data2/yangmrl/project/video2text/Worldsense_eval/results/plot/attn.png",dpi=300,bbox_inches='tight')
    plt.close()
    return

def get_lr(args,processor,input_ids):
    v_lst = []
    a_lst = []
    v_l = 1e9
    v_r = 0
    a_l = 1e9
    a_r = 0
    if 'qwen2.5_omni' in args.model_path:
        a = '<|AUDIO|>'
        v = '<|VIDEO|>'
    elif 'qwen3_omni' in args.model_path:
        a = '<|audio_pad|>'
        v = '<|video_pad|>'
    for i in range(len(input_ids)):
        # print(processor.decode(input_ids[i]))
        if processor.decode([input_ids[i]]) == v:
           v_lst.append(i)
           v_l = min(v_l,i)
           v_r = max(v_r,i)
        elif processor.decode([input_ids[i]]) == a:
           a_lst.append(i)
    return v_lst,a_lst

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

def calculate_sim(embd_tokens,video_tokens,audio_tokens):
    return

def show_frames(frames, num=32):
    plt.figure(figsize=(15,5))
    for i in range(min(num,len(frames))):
        plt.subplot(1, num, i+1)
        plt.imshow(frames[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("/mnt/data2/yangmrl/project/video2text/Worldsense_eval/results/plot/video_frames_qframe.png",dpi=300,bbox_inches='tight')
    plt.close()
    
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
    parser = argparse.ArgumentParser(description="inference on WorldSense benchmark")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument('--worldsense_json', type=str, required=True, help="WorldSense JSON path")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default="./worldsense_results")
    parser.add_argument('--mode', type=str, default="all", choices=["all", "video", "audio"])

    args = parser.parse_args()

    dataset = WorldSenseDataset(
        json_path=args.worldsense_json,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Loading model: {args.model_path}")
    model, processor = _load_model(args.model_path)
    actual_model = model  

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:2")
    model_dtype = actual_model.dtype
    cnt = 0
    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(dataloader):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):
            cnt+=1
            # if cnt<=1631:
            #     continue
            video_id = meta["video_id"]
            task_name = meta["task_name"]
            # if video_id != 'BQDKdEep':
            #     continue
            print(f"Processing WorldSense {video_id} | task={task_name} ...")
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

            
            video_pth = '/mnt/data2/yangmrl/project/video2text/test_data/worldsense/videos/'
            visual, frame_idx, frame_time, video_time = load_video(video_pth+video_id+".mp4", NFRAMES)
            
            
            try:
                indices = TextImageMatching(meta['question'], visual,tau=0.8)
            
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
            
            show_frames(visual)
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
            audios, images, videos = process_mm_info(new_message, use_audio_in_video=False)
             
            # print(audios[0])
            # video_tensor = videos[0]

            # frame0 = video_tensor[0]          # (3, H, W)
            # img = frame0.permute(1, 2, 0)      # → (H, W, C)
            # plt.imshow(img.cpu().numpy())
            # plt.axis('off')
            # plt.savefig("/mnt/data2/yangmrl/project/video2text/Worldsense_eval/results/plot/video_frames_grid.png",dpi=300)
            # plt.show()
                
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=False,
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
                        )
                    
                    hidden_states = outputs.hidden_states   
                    attention_weight = outputs.attentions
                    # attn_lstlayer = attention_weight[-1]
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = next_token_logits.argmax(dim=-1, keepdim=True).to(device)
                
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=1).to(device)
                    past_key_values = outputs.past_key_values
                    # past_key_values_audio = outputs_audio.past_key_values
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
                "video_id": meta["video_id"],
                "task_name": meta["task_name"],
                "question": meta["question"],
                "candidates": meta["candidates"],
                "gt_answer": meta["gt_answer"],
                "prediction": response,
                "raw_response": response,
                "domain": meta["domain"],
                "sub_category": meta["sub_category"],
                "video_duration": meta["video_duration"],
            }

            save_path = os.path.join(args.output_dir, f"{video_id}_{task_name}.json")
            with open(save_path, "w", encoding="utf-8") as fw:
                json.dump(result, fw, ensure_ascii=False, indent=2)

            print(f"{video_id} | {task_name} | pred: {response} | gt: {meta['gt_answer']}")

            del inputs, generated_ids, outputs
            torch.cuda.empty_cache()
            gc.collect()

    print("WorldSense inference completed:", args.output_dir)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
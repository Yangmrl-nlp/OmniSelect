import torch
import json
import os
import argparse
import gc
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from qwen_omni_utils import process_mm_info
from torch.utils.data import Dataset, DataLoader
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import seaborn as sns
import numpy as np
import decord
from PIL import Image
import random
import base64
import glob
import soundfile as sf
import librosa
import simplejpeg
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch.nn as nn
from tqdm import tqdm
import math

torch.set_grad_enabled(False)

# derived from ESResNeXt
SAMPLE_RATE = 16000
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

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
NFRAMES = 32
TOTAL_PIXELS =  NFRAMES * 768 * 28 * 28
MAX_NEW_TOKENS = 512

import sys
sys.path.append("/mnt/data2/yangmrl/project/video2text/AudioCLIP-master/")

from model import AudioCLIP
from utils.transforms import ToTensor1D
device = torch.device("cuda:2")
aclp = AudioCLIP(pretrained='/mnt/data2/yangmrl/project/video2text/AudioCLIP-master/assets/AudioCLIP-Full-Training.pt')
aclp.to(device).eval()
audio_transforms = ToTensor1D()
image_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        tv.transforms.CenterCrop(IMAGE_SIZE),
        tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
])

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
        options_str = "\n".join([c for c in candidates])

        prompt = TEST_PROMPT_WORLDSENSE.format(
            question=question,
            options_str=options_str
        )

        video_path = os.path.join(self.video_root, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        message = [
            dict(type='video', value=video_path),
            dict(type='text', value=prompt),
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

def get_lr(args,processor,input_ids):
    v_lst = []
    a_lst = []
    v_l = 1e9
    v_r = 0
    a_l = 1e9
    a_r = 0
    cnt_v = 0

    if 'qwen2.5_omni' in args.model_path:
        a = '<|AUDIO|>'
        v = '<|VIDEO|>'
    elif 'qwem3_omni' in args.model_path:
        a = '<|audio_pad|>'
        v = '<|video_pad|>'
    for i in range(len(input_ids)):
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
        fps = round(vr.get_avg_fps() /fps)
        
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

def show_frames(frames, num=5):
    plt.figure(figsize=(15,10))
    for i in range(min(num,len(frames))):
        plt.subplot(1, num, i+1)
        plt.imshow(frames[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("/mnt/data2/yangmrl/project/video2text/Worldsense_eval/results/plot/video_frames_qframe.png",dpi=300,bbox_inches='tight')
    plt.close()


def TextImageAudioMatching(processor, question, images, path_to_audio, nframes, topk=32):
    
    tokens = word_tokenize(question)
    tags = pos_tag(tokens)
    keywords = [word for word, tag in tags if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']]
    text_input = keywords if keywords else tokens

    track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
    track_len = len(track)
    chunk_size = track_len // nframes
    
    effective_len = nframes * chunk_size
    audio_chunks = torch.from_numpy(track[:effective_len]).view(nframes, 1, -1).to(device)
    
    images_tensor = torch.stack([image_transforms(img) for img in images]).to(device)

    with torch.no_grad():
        ((_, _, text_features), _), _ = aclp(text=text_input)
        text_features = F.normalize(text_features, dim=-1)

        # 获取音频特征
        ((audio_features, _, _), _), _ = aclp(audio=audio_chunks)
        audio_features = F.normalize(audio_features, dim=-1)

        # 获取图像特征
        ((_, image_features, _), _), _ = aclp(image=images_tensor)
        image_features = F.normalize(image_features, dim=-1)

        # 5. 计算 Logits
        scale_at = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
        scale_it = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)

        # 计算相似度矩阵并对文本维度取平均
        # [nframes, D] @ [D, n_words] -> [nframes, n_words] -> [nframes]
        logits_audio = (audio_features @ text_features.T).mean(dim=-1) * scale_at
        logits_image = (image_features @ text_features.T).mean(dim=-1) * scale_it

        actual_topk = min(topk, nframes)
        values_a, indices_a = torch.topk(logits_audio, actual_topk)
        values_v, indices_v = torch.topk(logits_image, actual_topk)
        
        a_score = logits_audio.mean()
        v_score = logits_image.mean()

    return indices_a, indices_v, a_score, v_score, chunk_size, logits_image, logits_audio

def calculate_time_group(video_id,T,h,w,temporal_patch_size = 2,fps = 2):
    
    group_v = video_id//(h*w)
    current_frame_id = group_v*(temporal_patch_size) 
    time_v = current_frame_id/fps 
    return time_v, group_v,current_frame_id

def generate_video_masks_np(t, h, w, num_blocks=8, block_size=30):

    masks = np.zeros((t, h, w), dtype=np.uint8)
    
    for i in range(t):
        for _ in range(num_blocks):
            y = np.random.randint(0, max(1, h - block_size))
            x = np.random.randint(0, max(1, w - block_size))
            masks[i, y : y + block_size, x : x + block_size] = 255
    return masks

def overlay_grey_blocks(img_np, mask_np, alpha=0.5):

    res_img = img_np.copy()
    mask_bool = mask_np > 0
    
    if np.any(mask_bool):
        grey_color = np.array([60, 60, 60], dtype=np.uint8)
        
        roi = res_img[mask_bool]
        blended = (roi.astype(np.float32) * (1 - alpha) + 
                   grey_color.astype(np.float32) * alpha).astype(np.uint8)
        res_img[mask_bool] = blended
    return res_img
    
def main():
    parser = argparse.ArgumentParser(description="inference on WorldSense benchmark")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument('--worldsense_json', type=str, required=True, help="WorldSense JSON path")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default="./worldsense_results")
    parser.add_argument('--mode', type=str, default="all", choices=["all", "video", "audio"])
    parser.add_argument('--cross_attn_ckpt_a', type=str, default="/mnt/data2/yangmrl/project/video2text/cross_attn_ckpts_a/13900.pt")
    parser.add_argument('--cross_attn_ckpt_v', type=str, default="/mnt/data2/yangmrl/project/video2text/cross_attn_ckpts_v/10000.pt")
    parser.add_argument('--prune_ratio_a', type=float, default=0.55)
    parser.add_argument('--prune_ratio_v', type=float, default=0.55)
    parser.add_argument('--prune', type=bool, default=True)
    args = parser.parse_args()

    dataset = WorldSenseDataset(
        json_path=args.worldsense_json,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Loading model: {args.model_path}")
    model, processor = _load_model(args.model_path)
    actual_model = model  
    actual_model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:2")
    model_dtype = actual_model.dtype
    cnt = 0
    nframes = 32
    fl = 1
    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(tqdm(dataloader, desc="OmniSelect Inference", total=len(dataloader))):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):
            cnt+=1
            video_id = meta["video_id"]
            task_name = meta["task_name"]
            
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
                elif item['type'] == 'video':
                    content.append({
                        'type': 'video', 'video': item['value'],
                        'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS,
                        'total_pixels': TOTAL_PIXELS, 'max_frames': NFRAMES, 
                        'video_start':0,
                        'video_end': seconds,
                    })
            
            video_pth = '/mnt/data2/yangmrl/project/video2text/test_data/worldsense/videos/'
            audio_pth = '/mnt/data2/yangmrl/project/video2text/test_data/worldsense/audios/'
            # visual, frame_idx, frame_time, video_time = load_video(video_pth+video_id+".mp4", NFRAMES)
            audio_pth += video_id+".wav"

            new_message = []
            new_message.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            })
            
            new_message.append({'role': 'user', 'content': content})
            text = processor.apply_chat_template(new_message, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(new_message, use_audio_in_video=True)
    
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=False,
                use_audio_in_video=True
            )
            
            # x1, y1 = 50, 30
            # x2, y2 = 150, 130
            # videos[0][:, :3, y1:y2, x1:x2] = 240
            
            numframes,_,h,w = videos[0].shape
            visual = (
                torch.nn.functional.interpolate(videos[0], size=(364,644))
                .permute(0,2,3,1)
                .cpu()
                .numpy()
                .astype("uint8")
            )
            # visual, frame_idx, frame_time, video_time = load_video(video_pth+video_id+".mp4", NFRAMES)
            # visual = visual[:videos[0].shape[0]]
            
            # videos_tmp = visual
            # mask_np = generate_video_masks_np(numframes,h,w)
            # show_frames(overlay_grey_blocks(videos_tmp,mask_np))
            
            print("Calculating Similarity Score...")
            indices_a,indices_v,a_score,v_score,CHUNK_SIZE,logits_v,logits_a = TextImageAudioMatching(processor, meta['question'], visual, audio_pth, min(nframes,seconds))
            values_a,_ = torch.sort(indices_a)
            values_v,_ = torch.sort(indices_v)
            video_first = a_score < v_score
            
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=model_dtype)
                    
            v_lst,a_lst = get_lr(args,processor,inputs['input_ids'][0]) # 26 * 46; 32 patches merge
            
            chunk = 0
            token_id2group = defaultdict(int)
            for i in range(len(v_lst)-1):
                token_id2group[v_lst[i]] = chunk+1
                if v_lst[i+1]-v_lst[i]!=1:
                    chunk+=1
            token_id2group[v_lst[-1]] = chunk+1 #chunk + 1个timegroup
            
            chunk = 0
            for i in range(len(a_lst)-1):
                token_id2group[a_lst[i]] = chunk+1
                if a_lst[i+1]-a_lst[i]!=1:
                    chunk+=1
            token_id2group[a_lst[-1]] = chunk+1 #chunk + 1个timegroup
            
            
            prune_need = {
                "logits_a" : logits_a,
                "logits_v" : logits_v,
                "video_first" : video_first,
                "args" : args,
                "token_id2group": token_id2group,
                "v_lst" : v_lst,
                "a_lst" : a_lst
            }
            
            #只选择关键帧和对应语音块
            # if not video_first:
            #     videos[0] = videos[0][values_a]
            #     audio_indices = values_a
            # else:
            #    videos[0] = videos[0][values_v]
            #    audio_indices = values_v
            
            # audios_new = list()
            # mp = defaultdict(bool)
            
            # for t in audio_indices:
            #     mp[int(t)] = True
            
            # for i in range(len(audios[0])):
            #     chunk_idx = i // CHUNK_SIZE
            #     if not mp[chunk_idx]:
            #         # audios_new.append(audios[0][i])
            #         audios[0][i] = 0
            # # audios[0]= np.array(audios_new)
            
            with torch.no_grad():
                # past_key_values = None
                # past_key_values_audio = None
                # generated_ids = inputs['input_ids'].clone().to(device)
                # new_token_list = []
                
                # # crossattn_model = CrossAttention(dim=2048, hidden=1024).to(device)
                # # state_dict = torch.load(args.cross_attn_ckpt, map_location='cpu')
                # # crossattn_model.load_state_dict(state_dict, strict=True)
                # # crossattn_model.eval()
                
                # for step in range(MAX_NEW_TOKENS):
                #     if step == 0:
                #         outputs = actual_model.thinker(
                #             **inputs,
                #             output_attentions=False,  
                #             past_key_values=past_key_values,
                #             use_cache=True,
                #             output_hidden_states=False,
                #             return_dict=True,
                #             use_audio_in_video=True,
                #             prune_need = prune_need
                #         )
                #         # audio_feature = extracted_states.get("audio_penultimate").float().to(device)
                #         # vision_feature = extracted_states.get("vision_penultimate").float().to(device)
                #         # if video_first:
                #         #     attn_map = crossattn_model(audio_feature, vision_feature)
                #         #     print(attn_map.shape)   
                #         # else:
                #         #     attn_map = crossattn_model(vision_feature,audio_feature)
                #         #     print(attn_map.shape)   
                #     else:
                #         current_inputs = {
                #             'input_ids': inputs['input_ids'].to(device),
                #             'attention_mask': inputs['attention_mask'].to(device),
                #         }
                        
                #         outputs = actual_model.thinker(
                #             **current_inputs,
                #             output_attentions=False,
                #             past_key_values=past_key_values,
                #             use_cache=True,
                #             output_hidden_states=False,
                #             return_dict=True,
                #             use_audio_in_video=True
                #         )
                    
                #     next_token_logits = outputs.logits[:, -1, :]
                #     next_token_id = next_token_logits.argmax(dim=-1, keepdim=True).to(device)
                    
                #     generated_ids = torch.cat([generated_ids, next_token_id], dim=1).to(device)
                #     past_key_values = outputs.past_key_values
                #     # past_key_values_audio = outputs_audio.past_key_values
                #     new_token_list.append(next_token_id.item())
                    
                #     attention_mask = torch.cat(
                #         [inputs['attention_mask'], torch.ones((inputs['attention_mask'].shape[0], 1), dtype=inputs['attention_mask'].dtype, device=device)],
                #         dim=-1
                #     )
                #     inputs['attention_mask'] = attention_mask
                #     inputs['input_ids'] = next_token_id
                #     if next_token_id.item() == processor.tokenizer.eos_token_id:
                #          break
                generated_ids = model.generate(
                        **inputs, 
                        thinker_max_new_tokens = 2, 
                        use_audio_in_video=True, 
                        return_audio=False, 
                        temperature=1,
                        prune_need = prune_need,
                    )
            prompt_length = inputs["input_ids"].shape[1]
            response = processor.tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)

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

            del inputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()

    print("WorldSense inference completed:", args.output_dir)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
    

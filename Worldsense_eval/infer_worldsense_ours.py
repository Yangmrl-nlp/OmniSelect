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

from PIL import Image
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

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
NFRAMES = 128
TOTAL_PIXELS =  NFRAMES * 768 * 28 * 28
MAX_NEW_TOKENS = 512

import sys
sys.path.append("/mnt/data2/yangmrl/project/video2text/AudioCLIP-master/")

from model import AudioCLIP
from utils.transforms import ToTensor1D

device = torch.device("cuda:3")

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
           cnt_v +=1
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


def show_frames(frames, num=32):
    plt.figure(figsize=(15,5))
    for i in range(min(num,len(frames))):
        plt.subplot(1, num, i+1)
        plt.imshow(frames[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("/mnt/data2/yangmrl/project/video2text/Worldsense_eval/results/plot/video_frames_qframe.png",dpi=300,bbox_inches='tight')
    plt.close()

def TextImageAudioMatching(processor,question, images,path_to_audio,nframes,topk = 32):
    tokens = word_tokenize(question)
    tags = pos_tag(tokens)
    text = [word for word, tag in tags if tag in ['NN', 'NNS', 'NNP', 'NNPS','JJ']]
    if len(text) == 0:
        text = tokens
    
    aclp = AudioCLIP(pretrained='/mnt/data2/yangmrl/project/video2text/AudioCLIP-master/assets/AudioCLIP-Full-Training.pt')
    aclp.eval()
    audio_transforms = ToTensor1D()

    image_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        tv.transforms.CenterCrop(IMAGE_SIZE),
        tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
    ])
    info = sf.info(path_to_audio)
    # duration = librosa.get_duration(path=path_to_audio, sr=SAMPLE_RATE)
    audio = list()
    track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
    
    spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
    pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
    
    CHUNK_SIZE = len(track) // (NFRAMES)
    
    for start in range(0, len(track), CHUNK_SIZE):
        end = start + CHUNK_SIZE
        if end > len(track): 
            
            pad_size = CHUNK_SIZE - (len(track) - start)
            x = track[start:len(track)]
            padded_x = np.pad(x, (0, pad_size), mode='constant', constant_values=0)
            audio.append((padded_x,None))
            break 
        
        chunk = track[start:end]
        pow_spec = None
        audio.append((chunk,pow_spec))
        start += CHUNK_SIZE
        
    audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
    images = torch.stack([image_transforms(image) for image in images])
    
    with torch.no_grad():
        ((audio_features, _, _), _), _ = aclp(audio=audio)
        ((_, image_features, _), _), _ = aclp(image=images)
        ((_, _, text_features), _), _ = aclp(text=text)
        
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
        image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
        text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)
        scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
        scale_image_text = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)
        logits_audio_text = scale_audio_text * audio_features @ text_features.T
        logits_image_text = scale_image_text * image_features @ text_features.T

        logits_audio_text = torch.mean(logits_audio_text,dim = -1)
        topk = min(topk,nframes)
        values_a,indices_a = torch.topk(logits_audio_text,topk,dim = -1)
        logits_image_text = torch.mean(logits_image_text,dim = -1)
        values_v,indices_v = torch.topk(logits_image_text,topk,dim = -1)
        a_score = torch.mean(logits_audio_text,dim = -1)
        v_score = torch.mean(logits_image_text,dim = -1)
        print(a_score,v_score)
    return indices_a,indices_v,a_score,v_score, CHUNK_SIZE,logits_image_text,logits_audio_text

def calculate_time_group(video_id,T,h,w,temporal_patch_size = 2,fps = 2):
    
    group_v = video_id//(h*w)
    current_frame_id = group_v*(temporal_patch_size) 
    time_v = current_frame_id/fps 
    return time_v, group_v,current_frame_id

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

    device = torch.device("cuda:3")
    model_dtype = actual_model.dtype
    cnt = 0
    nframes = 32
    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(dataloader):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):
            cnt+=1
            
            video_id = meta["video_id"]
            task_name = meta["task_name"]
            
            # if video_id != 'KQIbbWyN' or task_name != 'task1': 
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

            new_message = [{'role': 'user', 'content': content}]
            text = processor.apply_chat_template([new_message], tokenize=False, add_generation_prompt=True)
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
            
            visual = (
                torch.nn.functional.interpolate(videos[0], size=(364,644))
                .permute(0,2,3,1)
                .cpu()
                .numpy()
                .astype("uint8")
            )
            show_frames(visual)

            indices_a,indices_v,a_score,v_score,CHUNK_SIZE,logits_v,logits_a = TextImageAudioMatching(processor, meta['question'], visual, audio_pth, min(nframes,seconds))
            values_a,_ = torch.sort(indices_a)
            values_v,_ = torch.sort(indices_v)
            
            video_first = a_score < v_score

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=model_dtype)
            v_lst,a_lst = get_lr(args,processor,inputs['input_ids'][0]) #26 * 46; 32 patches merge
            actual_model.eval()
            chunk = 0
            for i in range(len(v_lst)-1):
                if v_lst[i+1]-v_lst[i]!=1:
                    chunk+=1
            # for i in range(len(a_lst)-1):
            #     if a_lst[i+1]-a_lst[i]!=1:
            #         print(i+1)
            # print(len(v_lst),len(a_lst),chunk+1)
            if not video_first:
                videos[0] = videos[0][values_a]
                audio_indices = values_a
            else:
               videos[0] = videos[0][values_v]
               audio_indices = values_v
            
            audios_new = list()
            mp = defaultdict(bool)
            
            for t in audio_indices:
                mp[int(t)] = True
            
            for i in range(len(audios[0])):
                chunk_idx = i // CHUNK_SIZE
                if not mp[chunk_idx]:
                    # audios_new.append(audios[0][i])
                    audios[0][i] = 0
            # audios[0]= np.array(audios_new)
            
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
                            use_audio_in_video=True
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
                            use_audio_in_video=True
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
    

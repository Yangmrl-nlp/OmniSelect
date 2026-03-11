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
TOTAL_PIXELS = NFRAMES * 768 * 28 * 28
MAX_NEW_TOKENS = 512


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

def calculate_time_group(video_id,T,h,w,temporal_patch_size = 2,fps = 2):
    
    group_v = video_id//(h*w)
    current_frame_id = group_v*(temporal_patch_size) 
    time_v = current_frame_id/fps 
    return time_v, group_v,current_frame_id

def attn_analyze(v_lst,a_lst,attn,layer_idx):
    num_heads, seq_len, _ = attn.shape
    attn = attn.squeeze(0)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attn.device))

    attn_causal = attn * causal_mask  # shape (num_heads, seq_len, seq_len)
    combined = torch.mean(attn_causal, dim = 0)  # shape (seq_len, seq_len)
    combined_ = torch.mean(combined,dim = 0)
    combined__ = []
    for i in range(len(combined_)):
        if i in v_lst:
            combined__.append(combined_[i])
    combined__ = torch.tensor(combined__)
    top_pct = 0.2
    topk = max(int(len(combined__) * top_pct), 1)  

    cnt = 0;cnt_ = 0
    values, indices = torch.topk(combined__, topk, dim=-1,sorted=True)
    for i in range(len(indices)):
        if indices[i] in v_lst:
            cnt+=1
        elif indices[i] in a_lst:
            cnt_+=1
    
    unique_top_ids = torch.unique(indices.flatten()).cpu().tolist()
    # print("Top token IDs:", unique_top_ids)
    entropy_layer_idx = -torch.sum(combined * torch.log(combined+1e-9), dim=-1)


    data = combined.cpu().numpy()

    mask = np.triu(np.ones_like(data, dtype=bool), k=1)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(data,mask=mask,cmap="viridis",square=True,xticklabels=False,yticklabels=False,vmin=1e-4,vmax=0.2)
    step = 100
    seq_len = data.shape[0]
    ticks = np.arange(0, seq_len, step)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(ticks, rotation=90)
    ax.set_yticklabels(ticks)

    plt.title("Combined Attention (all heads)")
    plt.savefig(
        f"/mnt/data2/yangmrl/project/video2text/Worldsense_eval/results/plot/attn_{layer_idx}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    return entropy_layer_idx, unique_top_ids

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

def calculate_sim(embd_tokens,video_tokens,audio_tokens):
    
    return

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
    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(dataloader):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):
            cnt+=1
            # if cnt<=2209:
            #     continue
            video_id = meta["video_id"]
            task_name = meta["task_name"]
            if video_id != 'GSLoYRyv':
                continue
            print(f"Processing WorldSense {video_id} | task={task_name} ...")
            duration = meta['video_duration']
            seconds = 0
            for t in duration:
                if t == 's':
                    continue
                seconds = seconds*10+int(t)
            
            content = []
            for item in message:
                
                if item['type'] == 'video':
                    content.append({
                        'type': 'video', 'video': item['value'],
                        'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS,
                        'total_pixels': TOTAL_PIXELS, 'max_frames': NFRAMES, 
                        'video_start':0,
                        'video_end': seconds
                    })
                elif item['type'] == 'text':
                    content.append({'type': 'text', 'text': item['value']})
                elif item['type'] == 'audio':
                    content.append({
                        'type': 'audio', 'audio': item['value'],
                        'audio_start':0,
                        'audio_end': seconds
                    })

            new_message = [{'role': 'user', 'content': content}]

            text = processor.apply_chat_template([new_message], tokenize=False, add_generation_prompt=True)
            
            audios, images, videos = process_mm_info(new_message,use_audio_in_video = False)

        
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
                            output_hidden_states=False,
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
                    attn_lstlayer = attention_weight[-1]


                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = next_token_logits.argmax(dim=-1, keepdim=True).to(device)
                    
                    v_lst,a_lst = get_lr(args,processor,inputs['input_ids'][0])
                    
                    # if step==0:
                    #     entropy = []
                    #     for i in range(len(attention_weight)):
                    #         # if i!=35:
                    #         #     continue
                    #         entropy_i,top_ids = attn_analyze(v_lst,a_lst,attention_weight[i][0],i)
                    #         entropy.append(entropy_i.mean().cpu())
                        
                    #     T = inputs['video_grid_thw'][0][0]
                    #     H = inputs['video_grid_thw'][0][1]
                    #     W = inputs['video_grid_thw'][0][2]
                    #     for idx in top_ids:
                    #         if idx in v_lst:
                    #             time_v, group_v,frame = calculate_time_group(idx,T,H/2,W/2)
                    #             # print(frame)
                            
                    #     plt.figure(figsize=(8, 4))
                    #     plt.plot(entropy, marker='o', linestyle='-')
                    #     plt.title("Average Attention Entropy Across Layers")
                    #     plt.xlabel("Layer")
                    #     plt.ylabel("Avg Attention Entropy")
                    #     plt.grid(True)
                    #     plt.savefig(
                    #     f"/mnt/data2/yangmrl/project/video2text/Worldsense_eval/results/plot/entropy_{video_id}.png",
                    #     dpi=300,
                    #     bbox_inches='tight')
                    #     plt.close()
                        
                    # print(processor.decode(inputs_audio['input_ids'][0]))
                    # print(next_token_id,next_token_id_audio)
                    # next_token_id = next_token_id.squeeze(0)
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
                    
                    # attention_mask_audio = torch.cat(
                    #     [inputs_audio['attention_mask'], torch.ones((inputs_audio['attention_mask'].shape[0], 1), dtype=inputs_audio['attention_mask'].dtype, device=device)],
                    #     dim=-1
                    # )
                    # inputs_audio['attention_mask'] = attention_mask_audio
                    # inputs_audio['input_ids'] = next_token_id

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

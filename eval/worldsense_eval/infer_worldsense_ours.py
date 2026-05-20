import torch
import json
import os
import argparse
import gc
from qwen_omni_utils import process_mm_info
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np
import time
import torchvision as tv
from PIL import Image
import sys
sys.path.append('/path/to/Omniselect/OmniSelect')
from transformers import Qwen2_5OmniProcessor
from modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

# derived from ESResNeXt
SAMPLE_RATE = 16000
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
MAX_NEW_TOKENS = 2
TEST_PROMPT_WORLDSENSE = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with **only the letter** (A, B, C, ...) of your choice.

Question: {question}
Options:
{options_str}
Answer:
"""

sys.path.append("/path/to/AudioCLIP/")

from model import AudioCLIP
from utils_a.transforms import ToTensor1D

aclp = AudioCLIP(pretrained='/path/to/AudioCLIP/assets/AudioCLIP-Full-Training.pt')
device = torch.device("cuda:0")
aclp.to(device).eval()
audio_transforms = ToTensor1D()
image_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        tv.transforms.CenterCrop(IMAGE_SIZE),
        tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
])

class WorldSenseDataset(Dataset):
    def __init__(self, json_path: str, video_root: str = "/path/to/worldsense/videos"):
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
            torch_dtype='auto',
            device_map='auto',
            attn_implementation='flash_attention_2'   
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
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

def TextImageAudioMatching(args,question,video_id, nframes,images, path_to_audio): 
    
    tokens = word_tokenize(question)
    tags = pos_tag(tokens)
    keywords = [word for word, tag in tags if (tag.startswith('NN') or tag.startswith('JJ'))]
    
    if len(keywords) == 0:
        text_input = tokens
    else:
        text_input = keywords 
    
    track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
    track_len = len(track)
    chunk_size = track_len // nframes
    
    effective_len = nframes * chunk_size
    audio_chunks = torch.from_numpy(track[:effective_len]).view(nframes, 1, -1).to(device)
    images_tensor = torch.stack([image_transforms(img) for img in images]).to(device)
    
    with torch.no_grad():
        ((audio_features, _, _), _), _ = aclp(audio=audio_chunks)
        ((_, image_features, _), _), _ = aclp(image=images_tensor)
        ((_, _, text_features), _), _ = aclp(text=text_input)
        
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
        image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
        text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

        scale_at = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
        scale_it = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)

        # [nframes, D] @ [D, n_words] -> [nframes, n_words] -> [nframes]
        logits_audio = (scale_at * audio_features @ text_features.T).mean(dim=-1)
        logits_image = (scale_it * image_features @ text_features.T).mean(dim=-1)
        
        a_score = logits_audio.mean()
        v_score = logits_image.mean()
    
    pre_a = a_score
    pre_v = v_score
    if abs(int(v_score) - int(a_score)) <= args.theta or len(keywords) == 0:
        a_score = v_score
    
    return a_score, v_score, logits_image, logits_audio, pre_a, pre_v

def show_frames(frames, num=5, output_dir="/mnt/data2/yangmrl/project/video2text/eval/Worldsense_eval/results/frames_plot"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(frames)):
        plt.figure(figsize=(5, 5)) 
        plt.imshow(frames[i])
        plt.axis("off")
        
        save_path = os.path.join(output_dir, f"video_frame_{i}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 

def get_win_cases(ours_dir, zip_dir):
    """
    自动对比并返回我们对、他们错的 Case 列表
    """
    ours_files = [f for f in os.listdir(ours_dir) if f.endswith('.json')]
    win_list = []
    for filename in ours_files:
        ours_path = os.path.join(ours_dir, filename)
        zip_path = os.path.join(zip_dir, filename)
        if not os.path.exists(zip_path): continue
        
        with open(ours_path, 'r') as f1, open(zip_path, 'r') as f2:
            d1, d2 = json.load(f1), json.load(f2)
            if d1['prediction'].strip().upper() == d1['gt_answer'].strip().upper() and \
               d2['prediction'].strip().upper() != d2['gt_answer'].strip().upper():
                win_list.append((d1['video_id'], d1['task_name']))
    return win_list

def main():
    parser = argparse.ArgumentParser(description="inference on WorldSense benchmark")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument('--worldsense_json', type=str, required=True, help="WorldSense JSON path")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default="./worldsense_results")
    parser.add_argument('--prune_ratio_a', type=float, default=0.40)
    parser.add_argument('--prune_ratio_v', type=float, default=0.60)
    parser.add_argument('--prune', type=bool, default=True)
    parser.add_argument('--nframes', type=int, default=32)
    parser.add_argument('--theta', type=float, default=3.7)
    parser.add_argument('--interactive', action='store_true', help="Enable interactive mode for case study")
    args = parser.parse_args()

    dataset = WorldSenseDataset(
        json_path=args.worldsense_json,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Loading model: {args.model_path}")
    model, processor = _load_model(args.model_path)
    actual_model = model  
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    NFRAMES = args.nframes
    TOTAL_PIXELS =  NFRAMES * 768 * 28 * 28
    cnt = 0
    fl = 1
    with open('/mnt/data2/yangmrl/project/video2text/eval/Worldsense_eval/results/case_study/temp.json') as f:
        dat = json.load(f)
    mp1 = defaultdict(bool)
    for i in dat:
        mp1[i['video_id']+i['task_name']] = True
    
    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(tqdm(dataloader, desc="OmniSelect Inference", total=len(dataloader))):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):

            video_id = meta["video_id"]
            task_name = meta["task_name"]
            
            if mp1[video_id+task_name]:
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
                if item['type'] == 'text':
                    content.append({'type': 'text', 'text': item['value']})
                elif item['type'] == 'video':
                    content.append({
                        'type': 'video', 'video': item['value'],
                        'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS,
                        'total_pixels': TOTAL_PIXELS, 'max_frames': NFRAMES, 
                        'video_start':0,
                        'video_end': seconds,
                    })
            
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
            nframes,c,h,w = videos[0].shape
            visual = (
                    torch.nn.functional.interpolate(videos[0], size=(h,w))
                    .permute(0,2,3,1)
                    .cpu()
                    .numpy()
                    .astype("uint8")
            )
            # show_frames(visual)
            
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=False,
                use_audio_in_video=True
            )
            a_score,v_score,logits_v,logits_a,pre_a,pre_v = TextImageAudioMatching(args, meta['question'], video_id, nframes,visual,audio_pth)
            print(a_score,v_score)
            if a_score < v_score:
                video_first = True
            elif a_score > v_score:
                video_first = False
            else:
                video_first = "uniform"
            inputs = inputs.to(model.device).to(model.dtype)
            v_lst,a_lst = get_lr(model.thinker.config.audio_token_id,model.thinker.config.video_token_id,inputs['input_ids'][0]) 
            
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
            if args.prune:
                prune_need = {
                    "logits_a" : logits_a,
                    "logits_v" : logits_v,
                    "video_first" : video_first,
                    "args" : args,
                    "token_id2group": token_id2group,
                    "v_lst" : v_lst,
                    "a_lst" : a_lst,
                    "video_id": video_id,    
                    "task_name": task_name ,  
                    "visual" : visual    
                }
            else:
                prune_need = None
            
            actual_model.eval()
            if hasattr(model, 'thinker'):
                model.thinker.nframes = videos[0].shape[0]
            
            torch.cuda.synchronize()
            start_time = time.time()

            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                generated_ids = model.generate(
                        **inputs, 
                        thinker_max_new_tokens = 2, 
                        use_audio_in_video=True, 
                        return_audio=False, 
                        temperature=1,
                        prune_need = prune_need,
                    )
            
            torch.cuda.synchronize()
            end_time = time.time()

            latency = end_time - start_time
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # MB
            # print(f"Latency: {latency:.4f} s | Peak GPU memory: {peak_mem:.2f} GB")
            
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
                "pre_abs": [float(pre_a),float(pre_v)],
                "gpu_mem": peak_mem
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
    

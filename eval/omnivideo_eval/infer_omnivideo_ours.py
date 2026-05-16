import torch
import json
import os
import re
import sys
import gc
import argparse
import warnings
from tqdm import tqdm
import librosa
import torchvision as tv
import numpy as np
import soundfile as sf
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
sys.path.append('/path/to/Omniselect/OmniSelect')
from modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from torch.utils.data import Dataset, DataLoader

# derived from ESResNeXt
SAMPLE_RATE = 16000
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
MAX_NEW_TOKENS = 2
TEST_PROMPT_OMNIVIDEO = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with **only the letter** (A, B, C, ...) of your choice.

Question: {question}
Options:
{options_str}
Answer:
"""

from utils.vision_process import process_vision_info
from utils.audio_process import process_audio_info
import utils.vision_process
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

def convert_duration_to_seconds(time_str):
    if not time_str or ':' not in str(time_str): return 0
    parts = str(time_str).split(':')
    try:
        if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except: return 0
    return 0
current_dir = os.path.dirname(os.path.abspath(__file__))

class OmniVideoDataset(Dataset):
    def __init__(self, json_path, data_root):
        self.data_root = data_root
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Succeed to loading {len(self.data)} entries from OmniVideoBench.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        video_rel_path = item.get('video') 
        video_id_clean = os.path.basename(video_rel_path).split('.')[0]
        
        question = item.get('question')
        candidates = item.get('options', [])
        options_str = "\n".join(candidates)
        
        prompt = TEST_PROMPT_OMNIVIDEO.format(
            question=question,
            options_str=options_str
        )
        
        video_full_path = os.path.join(self.data_root, video_rel_path)
        seconds = convert_duration_to_seconds(item.get('duration'))
        
        meta = {
            "video_id": video_id_clean,
            "question": question,
            "candidates": candidates,
            "gt_answer": item.get('correct_option'),
            "seconds": seconds,
            "question_type": item.get('question_type', 'N/A'),
            "idx": idx
        }
        return video_full_path, prompt, meta

def collate_fn(batch):
    return batch[0] 

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
    cha = abs(int(v_score) - int(a_score))
    if cha <= args.theta:
        theta = 0
    else:
        theta = 5
    if abs(v_score - a_score) <= args.theta or len(keywords) == 0:
        a_score = v_score
    
    return a_score, v_score, logits_image, logits_audio, pre_a, pre_v

def main():
    parser = argparse.ArgumentParser(description="Inference on OmniVideoBench")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--omnivideo_json', type=str, required=True, help="Path to OmniVideoBench data.json")
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of OmniVideoBench")
    parser.add_argument('--nframes', type=int, default=32, help="Number of frames to sample")
    parser.add_argument('--prune_ratio_a', type=float, default=0.40)
    parser.add_argument('--prune_ratio_v', type=float, default=0.60)
    parser.add_argument('--output_dir', type=str, default=os.path.join(current_dir, "results"))
    parser.add_argument('--prune', type=bool, default=True)
    parser.add_argument('--theta', type=float, default=3.7)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    print(f"Loading model: {args.model_path}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model.eval()

    dataset = OmniVideoDataset(args.omnivideo_json, args.data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    NFRAMES = args.nframes
    TOTAL_PIXELS = NFRAMES * 128 * 28 * 28
    all_results_list = [] 
    cnt = 0
    lst = []
    for video_path, prompt_text, meta in tqdm(dataloader, desc="Greedy Inference"):
        if not os.path.exists(video_path):
            continue

        video_id = meta["video_id"]
        sample_idx = meta["idx"]
        save_filename = f"{video_id}_q{sample_idx}.json"
        
        save_path = os.path.join(output_dir, save_filename)
        cnt+=1
    
        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    existing_result = json.load(f)
                    all_results_list.append(existing_result)
                continue 
            except Exception:
                pass 
        lst.append(video_id+str(sample_idx))
        try:
            content = [
                {
                    'type': 'video', 'video': video_path,
                    'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS,
                    'total_pixels': TOTAL_PIXELS, 'nframes': NFRAMES, 
                    'video_start': 0, 'video_end': meta['seconds'],
                },
                {'type': 'text', 'text': prompt_text}
            ]

            new_message = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
                },
                {"role": "user", "content": content}
            ]
            audio_pth = "/path/to/OmniVideoBench/audios/"
            audio_pth += video_id+".wav"
            
            text = processor.apply_chat_template(new_message, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_audio_info(new_message, True), *process_vision_info(new_message)
            nframes,c,h,w = videos[0].shape
            visual = (
                    torch.nn.functional.interpolate(videos[0], size=(h,w))
                    .permute(0,2,3,1)
                    .cpu()
                    .numpy()
                    .astype("uint8")
            )
            proc_kwargs = {
                "text": text,
                "audio": audios,
                "videos": videos,
                "return_tensors": "pt",
                "padding": False,
                "use_audio_in_video": True,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            }

            if images and len(images) > 0:
                proc_kwargs["images"] = images
        
            inputs = processor(**proc_kwargs).to(model.device).to(model.dtype)
            a_score,v_score,logits_v,logits_a,pre_a,pre_v = TextImageAudioMatching(args, meta['question'], video_id, nframes,visual,audio_pth)
            print(a_score,v_score)
            if a_score < v_score:
                video_first = True
            elif a_score > v_score:
                video_first = False
            else:
                video_first = "uniform"
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
                    "visual" : visual    
                }
            else:
                prune_need = None
                
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    thinker_max_new_tokens=2,  
                    use_audio_in_video=True, 
                    return_audio=False, 
                    temperature=1,
                    prune_need = prune_need,
                )
                    
            prompt_length = inputs["input_ids"].shape[1]
            response = processor.tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True).strip()

            match = re.search(r'\b([A-D])\b', response.upper())
            prediction = match.group(1) if match else response[:1]

            result = {
                "video_id": video_id,
                "sample_idx": sample_idx,
                "question": meta["question"],
                "candidates": meta["candidates"],
                "gt_answer": meta["gt_answer"],
                "prediction": prediction,
                "raw_response": response,
                "question_type": meta["question_type"],
                "is_correct": prediction.upper() == str(meta["gt_answer"]).upper()
            }

            save_filename = f"{video_id}_q{sample_idx}.json"
            save_path = os.path.join(output_dir, save_filename)
            with open(save_path, "w", encoding="utf-8") as fw:
                json.dump(result, fw, ensure_ascii=False, indent=2)

            all_results_list.append(result)

            if 'inputs' in locals(): del inputs
            if 'generated_ids' in locals(): del generated_ids
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error on {video_id}: {e}")

    summary_path = os.path.join(output_dir, "_all_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results_list, f, ensure_ascii=False, indent=2)

    total = len(all_results_list)
    correct = sum(1 for r in all_results_list if r['is_correct'])
    print(f"\nDone. Results saved in {output_dir}")
    print(f"Total Accuracy: {correct/total:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
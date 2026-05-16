import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import json
import gc
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from typing import Dict, List
from pathlib import Path
import torch.nn.functional as F
import torchvision as tv
import numpy as np
from PIL import Image
import librosa
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import sys
sys.path.append('/path/to/Omniselect/OmniSelect')
from modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
import soundfile as sf

torch.set_grad_enabled(False)

# derived from ESResNeXt
SAMPLE_RATE = 16000
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
TOTAL_PIXELS = 32 * 768 * 28 * 28
MAX_NEW_TOKENS = 512
TEST_PROMPT_DAILYOMNI = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with **only the letter** (A, B, C, ...) of your choice.

Question: {question}
Options:
{options_str}
Answer:
"""

sys.path.append("/path/to/AudioCLIP")
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

def _load_model(model_path, ):
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

    else:
        raise ValueError("Unsupported model")

    model.eval()
    return model, processor

def get_lr(audio_id, video_id, input_ids):
    input_ids = torch.as_tensor(input_ids)  
    v_lst = (input_ids == video_id).nonzero(as_tuple=True)[0]
    a_lst = (input_ids == audio_id).nonzero(as_tuple=True)[0]
    v_lst = v_lst.tolist()
    a_lst = a_lst.tolist()
    return v_lst, a_lst

def TextImageAudioMatching(args,question, nframes,images, path_to_audio): 
    
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
    
    if abs(int(v_score) - int(a_score)) <= args.theta or len(keywords) == 0:
        a_score = v_score
    
    return a_score, v_score, logits_image, logits_audio, pre_a, pre_v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--DailyOmni_json", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./dailyomni_eval/results")
    parser.add_argument('--nframes', type=int, default=32)
    parser.add_argument('--prune_ratio_a', type=float, default=0.40)
    parser.add_argument('--prune_ratio_v', type=float, default=0.60)
    parser.add_argument('--prune', type=bool, default=True)
    parser.add_argument('--theta', type=float, default=3.7)
    args = parser.parse_args()


    dataset = DailyOmniDataset(json_path=args.DailyOmni_json,video_root=args.video_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)
    print(f"Loading model: {args.model_path}")
    model, processor = _load_model(args.model_path)
    actual_model = model.module if hasattr(model, "module") else model
    NFRAMES = args.nframes
    TOTAL_PIXELS = NFRAMES * 768 * 28 * 28
    os.makedirs(args.output_dir, exist_ok=True)
    cnt = 0
    
    
    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(tqdm(dataloader, desc="OmniSelect Inference", total=len(dataloader))):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):
            cnt+=1
            qid = meta["qid"]
            video_id = meta['video_id']
            
            print(f"Processing DailyOmni qid = {qid} | video = {meta['video_id']}")
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
            
            
            audio_pth = args.video_root+f'/{video_id}/{video_id}'+'_audio.wav'
            new_message = []
            new_message.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            })
            new_message.append({'role': 'user', 'content': content})
            text = processor.apply_chat_template(new_message, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(new_message,use_audio_in_video = True)
            
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
            
            a_score,v_score,logits_v,logits_a,pre_a,pre_v = TextImageAudioMatching(args, meta['question'], nframes,visual,audio_pth)
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
            token_id2group[v_lst[-1]] = chunk+1 
            chunk = 0
            for i in range(len(a_lst)-1):
                token_id2group[a_lst[i]] = chunk+1
                if a_lst[i+1]-a_lst[i]!=1:
                    chunk+=1
            token_id2group[a_lst[-1]] = chunk+1 
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
            
            actual_model.eval()
            if hasattr(model, 'thinker'):
                model.thinker.nframes = videos[0].shape[0]
            
            with torch.no_grad():
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
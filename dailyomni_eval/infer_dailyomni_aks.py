import torch
import json
import os
import argparse
import gc
import heapq
import warnings
import numpy as np
import decord
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import base64
from io import BytesIO
from tqdm import tqdm  

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    CLIPProcessor,
    CLIPModel,
)
from qwen_omni_utils import process_mm_info

# =========================
# Global Config
# =========================

TEST_PROMPT_DAILYOMNI = """
Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.

Question: {question}
Choices:
{options_str}

Your answer should be a capital letter representing your choice: A, B, C, or D.
Don't generate any other text.
"""

AKS_MAX_FRAMES = 32
AKS_RATIO = 1
AKS_T1 = 0.8
AKS_T2 = -100
AKS_ALL_DEPTH = 5
MAX_NEW_TOKENS = 64

# =========================
# AKS Algorithm
# =========================

def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores, split_fn = [], []
    no_split_scores, no_split_fn = [], []
    for dic_score, fn in zip(dic_scores, fns):
        score, depth = dic_score["score"], dic_score["depth"]
        mean, std = np.mean(score), np.std(score)
        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            mid = len(score) // 2
            split_scores.append(dict(score=score[:mid], depth=depth + 1))
            split_scores.append(dict(score=score[mid:], depth=depth + 1))
            split_fn.append(fn[:mid]); split_fn.append(fn[mid:])
        else:
            no_split_scores.append(dic_score); no_split_fn.append(fn)
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score, all_split_fn = [], []
    return no_split_scores + all_split_score, no_split_fn + all_split_fn

def aks_select_frames(video_path, question, clip_model, clip_processor, device):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_frames = int(len(vr) / int(fps))
    
    # 1. 提取文本特征
    inputs_text = clip_processor(text=question, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs_text)
        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, "text_embeds"):
                text_features = text_features.text_embeds
            elif hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            else:
                text_features = text_features[0]
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 2. 逐秒计算相似度
    scores, frame_ids = [], []
    for j in range(total_frames):
        frame_idx = j * int(fps)
        frame = Image.fromarray(vr[frame_idx].asnumpy())
        inputs_img = clip_processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feat = clip_model.get_image_features(**inputs_img)
            if not isinstance(img_feat, torch.Tensor):
                if hasattr(img_feat, "image_embeds"):
                    img_feat = img_feat.image_embeds
                elif hasattr(img_feat, "pooler_output"):
                    img_feat = img_feat.pooler_output
                else:
                    img_feat = img_feat[0]
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
        
        sim = F.cosine_similarity(text_features, img_feat).item()
        scores.append(sim)
        frame_ids.append(frame_idx)

    if len(scores) < AKS_MAX_FRAMES: 
        return frame_ids

    # 3. 递归筛选 (后续逻辑保持不变)
    score_np = np.array(scores)
    normalized = (score_np - score_np.min()) / (score_np.max() - score_np.min() + 1e-8)
    a, b = meanstd(len(scores), [dict(score=normalized.tolist(), depth=0)], AKS_MAX_FRAMES, [frame_ids], AKS_T1, AKS_T2, AKS_ALL_DEPTH)
    selected = []
    for s, f in zip(a, b):
        k = int(AKS_MAX_FRAMES / (2 ** s["depth"]))
        topk = heapq.nlargest(k, range(len(s["score"])), s["score"].__getitem__)
        selected.extend([f[t] for t in topk])
    return sorted(selected)

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
        qid, video_id = idx, item["video_id"]
        options_str = "\n".join(item["Choice"])
        prompt = TEST_PROMPT_DAILYOMNI.format(question=item["Question"], options_str=options_str)
        v_path = os.path.join(self.video_root, video_id, f"{video_id}_video.mp4")
        a_path = os.path.join(self.video_root, video_id, f"{video_id}_audio.wav")
        message = [dict(type="text", value=prompt), dict(type="audio", value=a_path), dict(type="video", value=v_path)]
        return message, idx, {"qid": qid, "video_id": video_id, "question": item["Question"], "options": item["Choice"], "gt_letter": item["Answer"]}

def collate_fn(batch):
    messages, indices, metas = zip(*batch)
    return list(messages), list(indices), list(metas)

# =========================
# Model Loader
# =========================

def _load_model(model_path, device):
    if "qwen2.5_omni" in model_path.lower():
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    elif "qwen3_omni" in model_path.lower():
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto", attn_implementation="flash_attention_2",
            low_cpu_mem_usage=False #---防止异步加载模型cuda出错
        )
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
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(f"cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 CLIP
    clip_path = "/mnt/data2/yangmrl/project/video2text/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_path)
    # import sys
    # sys.path.append("/mnt/data2/yangmrl/project/video2text/Long-CLIP")
    # from model import longclip
    # model_path = "/mnt/data2/yangmrl/project/video2text/models/clip/longclip-B.pt"
    # clip_model, clip_processor = longclip.load(model_path, device=device)

    # 加载 Qwen-Omni
    model, processor = _load_model(args.model_path, device)
    actual_model = model

    dataset = DailyOmniDataset(args.daily_json, args.video_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    pbar = tqdm(total=len(dataset), desc="Processing DailyOmni")

    for messages_batch, indices, metas_batch in dataloader:
        for message, idx, meta in zip(messages_batch, indices, metas_batch):
            qid, video_id = meta["qid"], meta["video_id"]
            output_filename = os.path.join(args.output_dir, f"{video_id}_task{qid}.json")
            if os.path.exists(output_filename):
                print(f"Skipping {video_id} (already exists).") 
                pbar.update(1)
                continue

            video_path = next(m["value"] for m in message if m["type"] == "video")
            pbar.set_postfix({"video_id": video_id})

            print(f"Processing qid={qid} | video={video_id}")

            # 1. AKS 选帧
            selected_indices = aks_select_frames(video_path, meta["question"], clip_model, clip_processor, device)
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            visual = [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in selected_indices]

            # 2. 构造 Content 
            content = []
            for item in message:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["value"]})
                elif item["type"] == "audio" and os.path.exists(item["value"]):
                    content.append({"type": "audio", "audio": item["value"]})

            # Base64 循环：将 AKS 选出的帧一张张作为 image 塞进去
            for img in visual:
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
            
            new_message = [{"role": "user", "content": content}]
            
            text = processor.apply_chat_template([new_message], tokenize=False, add_generation_prompt=True)
            # 提取多模态特征 (process_mm_info 会自动把 Base64 转为 Tensor 放到 images 里)
            audios, images, videos = process_mm_info(new_message, use_audio_in_video=False)

            inputs = processor(
                text=text, audio=audios, images=images, videos=None,
                return_tensors="pt", padding=False, 
                use_audio_in_video=False
            )

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=actual_model.dtype)

            try:
                with torch.no_grad():
                    # generated_ids = actual_model.generate(
                    #     **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    #     pad_token_id=processor.tokenizer.eos_token_id,
                    #     use_cache=True, use_audio_in_video=False
                    # )

                    # if isinstance(generated_ids, (list, tuple)):
                    #     generated_ids = generated_ids[0]
                    # input_len = inputs["input_ids"].shape[1]
                    # new_token_ids = generated_ids[0][input_len:] 
                    
                    # response = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
                    
                    past_key_values = None
                    generated_ids = inputs['input_ids'].clone().to(device)
                    new_token_list = []

                    # 如果模型有 thinker 属性就用 thinker，没有就直接用 actual_model
                    inference_model = actual_model.thinker if hasattr(actual_model, 'thinker') else actual_model
                    #模拟自回归过程
                    for step in range(MAX_NEW_TOKENS):
                        if step == 0:
                            # 第一步，传入完整的多模态 inputs
                            outputs = inference_model(
                                **inputs,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                            )
                        else:
                            # 后续步骤，只传上一个 token 和 KV cache
                            current_inputs = {
                                'input_ids': next_token_id,
                                'attention_mask': inputs['attention_mask'],
                            }
                            outputs = inference_model(
                                **current_inputs,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                            )
                        
                        past_key_values = outputs.past_key_values
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
                    
                        if next_token_id.item() == processor.tokenizer.eos_token_id:
                            break
                            
                        new_token_list.append(next_token_id.item())
                        
                        # 更新 attention_mask 以包含新 token
                        inputs['attention_mask'] = torch.cat([
                            inputs['attention_mask'], 
                            torch.ones((1, 1), device=device, dtype=inputs['attention_mask'].dtype)
                        ], dim=-1)

                    response = processor.tokenizer.decode(new_token_list, skip_special_tokens=True).strip()


                result = {"qid": meta["qid"], "video_id": video_id, "prediction": response, "gt_letter": meta["gt_letter"]}
                with open(os.path.join(args.output_dir, f"{video_id}_task{qid}.json"), "w") as f:
                    json.dump(result, f, indent=2)
                
                print(f"{video_id} | Pred: {response} | GT: {meta['gt_letter']}")

            except Exception as e:
                print(f"Error processing {video_id}: {e}")

            del inputs, visual
            torch.cuda.empty_cache()
            gc.collect()
            pbar.update(1)

    pbar.close()
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# import argparse
# import json
# import gc
# import heapq
# import numpy as np
# import decord
# import base64
# from io import BytesIO
# from PIL import Image
# from tqdm import tqdm
# import logging 
# import torch.nn.functional as F
# import warnings

# from transformers import (
#     Qwen2_5OmniForConditionalGeneration,
#     Qwen2_5OmniProcessor,
#     Qwen3OmniMoeForConditionalGeneration,
#     Qwen3OmniMoeProcessor,
#     CLIPProcessor,
#     CLIPModel,
# )
# from qwen_omni_utils import process_mm_info

# # =========================
# # Global Constants
# # =========================
# TEST_PROMPT_DAILYOMNI = """
# Your task is to accurately answer multiple-choice questions based on the given video.
# Select the single most accurate answer from the given choices.

# Question: {question}
# Choices:
# {options_str}

# Your answer should be a capital letter representing your choice: A, B, C, or D.
# Don't generate any other text.
# """

# AKS_MAX_FRAMES = 32
# AKS_T1 = 0.8
# AKS_T2 = -100
# AKS_ALL_DEPTH = 5
# MAX_NEW_TOKENS = 64

# # =========================
# # AKS Algorithm Logic
# # =========================

# def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
#     split_scores, split_fn = [], []
#     no_split_scores, no_split_fn = [], []
#     for dic_score, fn in zip(dic_scores, fns):
#         score, depth = dic_score["score"], dic_score["depth"]
#         mean, std = np.mean(score), np.std(score)
#         top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
#         top_score = [score[t] for t in top_n]
#         mean_diff = np.mean(top_score) - mean
#         if mean_diff > t1 and std > t2:
#             no_split_scores.append(dic_score)
#             no_split_fn.append(fn)
#         elif depth < all_depth:
#             mid = len(score) // 2
#             split_scores.append(dict(score=score[:mid], depth=depth + 1))
#             split_scores.append(dict(score=score[mid:], depth=depth + 1))
#             split_fn.append(fn[:mid]); split_fn.append(fn[mid:])
#         else:
#             no_split_scores.append(dic_score); no_split_fn.append(fn)
#     if len(split_scores) > 0:
#         all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
#     else:
#         all_split_score, all_split_fn = [], []
#     return no_split_scores + all_split_score, no_split_fn + all_split_fn

# def aks_select_frames(video_path, question, clip_model, clip_processor, device):
#     vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
#     total_frames_in_video = len(vr)
    
#     # 提取文本特征
#     inputs_text = clip_processor(text=question, return_tensors="pt", truncation=True).to(device)
#     with torch.no_grad():
#         text_features = clip_model.get_text_features(**inputs_text)
#         # 兼容性修复：确保拿到的是 Tensor
#         if hasattr(text_features, "pooler_output"):
#             text_features = text_features.pooler_output
#         elif not isinstance(text_features, torch.Tensor):
#             text_features = text_features[0]
            
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     # 构造候选池
#     num_candidates = min(total_frames_in_video, 96) 
#     candidate_indices = np.linspace(0, total_frames_in_video - 1, num_candidates, dtype=int)

#     scores, frame_ids = [], []
#     for idx in candidate_indices:
#         frame = Image.fromarray(vr[idx].asnumpy())
#         inputs_img = clip_processor(images=frame, return_tensors="pt").to(device)
#         with torch.no_grad():
#             img_feat = clip_model.get_image_features(**inputs_img)
#             if hasattr(img_feat, "pooler_output"):
#                 img_feat = img_feat.pooler_output
#             elif not isinstance(img_feat, torch.Tensor):
#                 img_feat = img_feat[0]
                
#             img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
#         sim = F.cosine_similarity(text_features, img_feat).item()
#         scores.append(sim)
#         frame_ids.append(int(idx))

#     if len(scores) <= AKS_MAX_FRAMES: 
#         return sorted(frame_ids)

#     score_np = np.array(scores)
#     normalized = (score_np - score_np.min()) / (score_np.max() - score_np.min() + 1e-8)
#     a, b = meanstd(len(scores), [dict(score=normalized.tolist(), depth=0)], AKS_MAX_FRAMES, [frame_ids], AKS_T1, AKS_T2, AKS_ALL_DEPTH)
    
#     selected = []
#     for s, f in zip(a, b):
#         k = max(1, int(AKS_MAX_FRAMES / (2 ** s["depth"])))
#         topk = heapq.nlargest(k, range(len(s["score"])), s["score"].__getitem__)
#         selected.extend([f[t] for t in topk])
    
#     return sorted(list(set(selected)))[:AKS_MAX_FRAMES]

# # =========================
# # Dataset
# # =========================

# class DailyOmniDataset(Dataset):
#     def __init__(self, json_path, video_root):
#         self.video_root = video_root
#         with open(json_path, "r", encoding="utf-8") as f:
#             self.data = json.load(f)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         qid, video_id = idx, item["video_id"]
#         options = item["Choice"]
#         options_str = "\n".join(options)
#         prompt = TEST_PROMPT_DAILYOMNI.format(question=item["Question"], options_str=options_str)
#         v_path = os.path.join(self.video_root, video_id, f"{video_id}_video.mp4")
#         a_path = os.path.join(self.video_root, video_id, f"{video_id}_audio.wav")
        
#         return {
#             "prompt": prompt,
#             "v_path": v_path,
#             "a_path": a_path,
#             "meta": {
#                 "qid": qid,
#                 "video_id": video_id,
#                 "question": item["Question"],
#                 "options": options,
#                 "gt_letter": item["Answer"]
#             }
#         }

# def collate_fn(batch):
#     return batch

# # =========================
# # Main Evaluation Script
# # =========================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--daily_json", type=str, required=True)
#     parser.add_argument("--video_root", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, default="./results")
#     parser.add_argument("--gpu_id", type=int, default=0)
#     parser.add_argument("--num_gpus", type=int, default=4)
#     args = parser.parse_args()

#     # 1. 环境与日志
#     os.makedirs(args.output_dir, exist_ok=True)
#     log_dir = os.path.join(args.output_dir, "logs")
#     os.makedirs(log_dir, exist_ok=True)
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - GPU %(gpu)s - %(message)s',
#         handlers=[logging.FileHandler(os.path.join(log_dir, f"gpu_{args.gpu_id}.log"), encoding='utf-8')]
#     )
#     logger = logging.getLogger(__name__)
#     # 注入 gpu_id 到日志格式
#     logger = logging.LoggerAdapter(logger, {'gpu': args.gpu_id})

#     device = torch.device(f"cuda:{args.gpu_id}")
#     torch.cuda.set_device(device)

#     # 2. 加载模型
#     logger.info("Loading CLIP model...")
#     clip_path = "/mnt/data2/yangmrl/project/video2text/clip-vit-base-patch32"
#     clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()
#     clip_processor = CLIPProcessor.from_pretrained(clip_path)

#     logger.info(f"Loading Qwen-Omni from {args.model_path}")
#     if "qwen2.5_omni" in args.model_path.lower():
#         model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#             args.model_path, torch_dtype=torch.float16, device_map={"": device}, attn_implementation="flash_attention_2"
#         )
#         processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
#     elif "qwen3_omni" in args.model_path.lower():
#         model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
#             args.model_path, dtype="auto", device_map={"": device}, attn_implementation="flash_attention_2"
#         )
#         processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
#     model.eval()

#     # 3. 数据加载与分片
#     dataset = DailyOmniDataset(args.daily_json, args.video_root)
#     subset_indices = [i for i in range(len(dataset)) if i % args.num_gpus == args.gpu_id]
#     subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
#     dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

#     pbar = tqdm(total=len(subset_dataset), desc=f"GPU {args.gpu_id}", position=args.gpu_id)

#     # 4. 推理循环
#     for batch in dataloader:
#         data = batch[0]
#         meta = data["meta"]
#         qid, video_id = meta["qid"], meta["video_id"]
#         save_path = os.path.join(args.output_dir, f"{video_id}_task{qid}.json")

#         if os.path.exists(save_path):
#             pbar.update(1)
#             continue

#         try:
#             # --- Step A: AKS 选帧 ---
#             selected_indices = aks_select_frames(data["v_path"], meta["question"], clip_model, clip_processor, device)
#             vr = decord.VideoReader(data["v_path"], ctx=decord.cpu(0))
            
#             # --- Step B: 构造多模态消息 ---
#             content = [{"type": "text", "text": data["prompt"]}]
            
#             # 音频处理
#             if os.path.exists(data["a_path"]):
#                 content.append({"type": "audio", "audio": data["a_path"]})

#             # 视频帧转 Base64 塞入消息
#             for idx in selected_indices:
#                 img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
#                 buf = BytesIO()
#                 img.save(buf, format="JPEG")
#                 b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
#                 content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64_str}"})
            
#             # --- Step C: 模型生成 ---
#             new_message = [{"role": "user", "content": content}]
#             text = processor.apply_chat_template([new_message], tokenize=False, add_generation_prompt=True)
#             audios, images, videos = process_mm_info(new_message, use_audio_in_video=False)

#             inputs = processor(
#                 text=text, audio=audios, images=images, videos=None,
#                 return_tensors="pt", padding=False, use_audio_in_video=False
#             )
#             inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
#             # 统一 Data Type
#             for k, v in inputs.items():
#                 if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
#                     inputs[k] = v.to(dtype=model.dtype)

#             with torch.no_grad():
#                 generated_ids = model.generate(
#                     **inputs, max_new_tokens=MAX_NEW_TOKENS,
#                     pad_token_id=processor.tokenizer.eos_token_id,
#                     use_cache=True, use_audio_in_video=False
#                 )
                
#                 # 兼容不同版本的 generate 返回值类型
#                 if isinstance(generated_ids, tuple):
#                     generated_ids = generated_ids[0]
                
#                 input_len = inputs["input_ids"].shape[1]
#                 response = processor.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

#             # --- Step D: 保存结果 ---
#             result = {
#                 "qid": qid,
#                 "video_id": video_id,
#                 "question": meta["question"],
#                 "options": meta["options"],
#                 "gt_letter": meta["gt_letter"],
#                 "prediction": response,
#                 "raw_response": response
#             }
#             with open(save_path, "w", encoding="utf-8") as fw:
#                 json.dump(result, fw, ensure_ascii=False, indent=2)

#             logger.info(f"DONE: {video_id} | QID: {qid} | Pred: {response} | GT: {meta['gt_letter']}")

#         except Exception as e:
#             logger.error(f"FAILED: {video_id} | QID: {qid} | Error: {str(e)}")

#         # 显存深度清理
#         del content, new_message, inputs
#         if 'generated_ids' in locals(): del generated_ids
#         torch.cuda.empty_cache()
#         gc.collect()
#         pbar.update(1)

#     pbar.close()
#     logger.info(f"GPU {args.gpu_id} finished all tasks.")

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()
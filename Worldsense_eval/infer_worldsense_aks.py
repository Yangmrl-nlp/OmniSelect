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
# TEST_PROMPT_WORLDSENSE = """
# These are the frames of a video and the corresponding audio.
# Please answer the following multiple-choice question based on the video and audio content.
# Choose the correct option and respond with **only the letter** (A, B, C, ...) of your choice.

# Question: {question}
# Options:
# {options_str}
# Answer:
# """

# AKS_MAX_FRAMES = 32
# AKS_T1 = 0.8
# AKS_T2 = -100
# AKS_ALL_DEPTH = 5
# MAX_NEW_TOKENS = 512

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
    
#     # Text features
#     inputs_text = clip_processor(text=question, return_tensors="pt", truncation=True).to(device)
#     with torch.no_grad():
#         text_features = clip_model.get_text_features(**inputs_text)
#         if hasattr(text_features, "pooler_output"):
#             text_features = text_features.pooler_output
#         elif not isinstance(text_features, torch.Tensor):
#             text_features = text_features[0]
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     # Candidate pool
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
# # WorldSense Dataset
# # =========================

# # class WorldSenseDataset(Dataset):
# #     def __init__(self, json_path, video_root):
# #         self.video_root = video_root
# #         with open(json_path, 'r', encoding='utf-8') as f:
# #             data = json.load(f)

# #         self.samples = []
# #         for video_id, item in data.items():
# #             for task_name, task in item.items():
# #                 if task_name.startswith("task"):
# #                     self.samples.append({
# #                         "video_id": video_id,
# #                         "task_name": task_name,
# #                         "task_data": task,
# #                         "full_item": item
# #                     })

# #         self.samples = self.samples[:5]

# #     def __len__(self):
# #         return len(self.samples)
# class WorldSenseDataset(Dataset):
#     def __init__(self, json_path, video_root, gpu_id=0, num_gpus=1):
#         self.video_root = video_root
#         with open(json_path, 'r', encoding='utf-8') as f:
#             full_data = json.load(f)

#         # 1. 获取所有唯一的 video_id 并排序（排序保证 4 个进程拿到的顺序一致）
#         all_video_ids = sorted(list(full_data.keys()))
        
#         # 2. 按照 gpu_id 对视频进行分片
#         # 这样确保了同一个 video_id 只会出现在某一个进程中
#         my_video_ids = [vid for i, vid in enumerate(all_video_ids) if i % num_gpus == gpu_id]

#         # 3. 只拉平属于当前 GPU 的视频任务
#         self.samples = []
#         for vid in my_video_ids:
#             item = full_data[vid]
#             for task_name, task in item.items():
#                 if task_name.startswith("task"):
#                     self.samples.append({
#                         "video_id": vid,
#                         "task_name": task_name,
#                         "task_data": task,
#                         "full_item": item
#                     })
#         # self.samples = self.samples[:5]  # 仅用于测试，正式运行时注释掉

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         video_id = sample["video_id"]
#         task_name = sample["task_name"]
#         task = sample["task_data"]
        
#         question = task["question"]
#         candidates = task["candidates"]
#         alphas = [chr(65 + i) + ". " for i in range(len(candidates))]
#         options_str = "\n".join([a + c for a, c in zip(alphas, candidates)])

#         prompt = TEST_PROMPT_WORLDSENSE.format(
#             question=question,
#             options_str=options_str
#         )

#         v_path = os.path.join(self.video_root, f"{video_id}.mp4")
        
#         return {
#             "prompt": prompt,
#             "v_path": v_path,
#             "meta": {
#                 "video_id": video_id,
#                 "task_name": task_name,
#                 "question": question,
#                 "candidates": candidates,
#                 "gt_answer": task["answer"],
#                 "domain": sample["full_item"].get("domain", "unknown"),
#                 "sub_category": sample["full_item"].get("sub_category", "unknown"),
#             }
#         }

# def collate_fn(batch):
#     return batch

# # =========================
# # Main Script
# # =========================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--worldsense_json", type=str, required=True)
#     parser.add_argument("--video_root", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, default="./worldsense_aks_results")
#     parser.add_argument("--gpu_id", type=int, default=0)
#     parser.add_argument("--num_gpus", type=int, default=4)
#     args = parser.parse_args()

#     # 1. 环境与日志初始化
#     os.makedirs(args.output_dir, exist_ok=True)
#     log_dir = os.path.join(args.output_dir, "logs")
#     os.makedirs(log_dir, exist_ok=True)
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - GPU %(gpu)s - %(message)s',
#         handlers=[logging.FileHandler(os.path.join(log_dir, f"gpu_{args.gpu_id}.log"), encoding='utf-8')]
#     )
#     logger = logging.getLogger(__name__)
#     logger = logging.LoggerAdapter(logger, {'gpu': args.gpu_id})

#     device = torch.device(f"cuda:{args.gpu_id}")
#     torch.cuda.set_device(device)

#     # 2. 加载 CLIP 模型 (用于 AKS)
#     logger.info("Loading CLIP model...")
#     clip_path = "/mnt/data2/yangmrl/project/video2text/clip-vit-base-patch32"
#     clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()
#     clip_processor = CLIPProcessor.from_pretrained(clip_path)

#     # 3. 加载 Qwen-Omni 模型
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

#     # 4. 数据加载 (内部已按视频 ID 进行多卡分片)
#     dataset = WorldSenseDataset(
#         args.worldsense_json, 
#         args.video_root, 
#         gpu_id=args.gpu_id, 
#         num_gpus=args.num_gpus
#     )
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     # --- 视频缓存变量 ---
#     last_video_id = None
#     cached_selected_indices = None
#     cached_images_b64 = []  # 存储 base64 字符串列表
#     # ------------------

#     pbar = tqdm(total=len(dataset), desc=f"GPU {args.gpu_id}", position=args.gpu_id)

#     # 5. 推理循环
#     for batch in dataloader:
#         data = batch[0]
#         meta = data["meta"]
#         v_id, t_name = meta["video_id"], meta["task_name"]
#         save_path = os.path.join(args.output_dir, f"{v_id}_{t_name}.json")

#         # 检查是否已处理
#         if os.path.exists(save_path):
#             pbar.update(1)
#             continue

#         try:
#             # --- Step A: AKS 选帧与缓存逻辑 ---
#             if v_id == last_video_id:
#                 # 命缓存：复用上一个 task 的选帧和图像数据
#                 selected_indices = cached_selected_indices
#                 images_b64 = cached_images_b64
#                 # logger.info(f"Using cached frames for {v_id} | {t_name}")
#             else:
#                 # 未命中：重新运行 AKS 选帧
#                 selected_indices = aks_select_frames(data["v_path"], meta["question"], clip_model, clip_processor, device)
#                 vr = decord.VideoReader(data["v_path"], ctx=decord.cpu(0))
                
#                 # 预先处理图像并转为 Base64 缓存
#                 images_b64 = []
#                 for idx in selected_indices:
#                     img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
#                     buf = BytesIO()
#                     img.save(buf, format="JPEG")
#                     b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
#                     images_b64.append(b64_str)
                
#                 # 更新缓存状态
#                 last_video_id = v_id
#                 cached_selected_indices = selected_indices
#                 cached_images_b64 = images_b64
#             # --------------------------------

#             # --- Step B: 构造多模态消息 ---
#             content = [{"type": "text", "text": data["prompt"]}]
            
#             # 添加音频 (复用视频文件路径)
#             content.append({"type": "audio", "audio": data["v_path"]})

#             # 添加 AKS 选出的帧 (作为 Image 类型传入)
#             for b64_data in images_b64:
#                 content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64_data}"})
            
#             new_message = [{"role": "user", "content": content}]
#             text = processor.apply_chat_template([new_message], tokenize=False, add_generation_prompt=True)
            
#             # 处理多模态张量 (AKS 模式下 videos 为空，使用 images)
#             audios, images, videos = process_mm_info(new_message, use_audio_in_video=False)

#             inputs = processor(
#                 text=text, audio=audios, images=images, videos=None,
#                 return_tensors="pt", padding=False, use_audio_in_video=False
#             )
#             inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
#             # 确保 Data Type 一致
#             for k, v in inputs.items():
#                 if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
#                     inputs[k] = v.to(dtype=model.dtype)

#             # --- Step C: 模型生成 ---
#             with torch.no_grad():
#                 generated_ids = model.generate(
#                     **inputs, 
#                     max_new_tokens=MAX_NEW_TOKENS,
#                     pad_token_id=processor.tokenizer.eos_token_id,
#                     use_cache=True,
#                     use_audio_in_video=False
#                 )
                
#                 if isinstance(generated_ids, tuple):
#                     generated_ids = generated_ids[0]
                
#                 input_len = inputs["input_ids"].shape[1]
#                 response = processor.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

#             # --- Step D: 保存结果 ---
#             result = {
#                 "video_id": v_id,
#                 "task_name": t_name,
#                 "question": meta["question"],
#                 "candidates": meta["candidates"],
#                 "gt_answer": meta["gt_answer"],
#                 "prediction": response,
#                 "raw_response": response,
#                 "domain": meta["domain"],
#                 "sub_category": meta["sub_category"],
#                 "aks_selected_indices": selected_indices # 记录选了哪些帧
#             }
#             with open(save_path, "w", encoding="utf-8") as fw:
#                 json.dump(result, fw, ensure_ascii=False, indent=2)

#             logger.info(f"DONE: {v_id} | {t_name} | Pred: {response} | GT: {meta['gt_answer']}")

#         except Exception as e:
#             logger.error(f"FAILED: {v_id} | {t_name} | Error: {str(e)}")
#             import traceback
#             # logger.error(traceback.format_exc())

#         # 深度清理显存
#         del content, new_message, inputs
#         if 'generated_ids' in locals(): del generated_ids
#         torch.cuda.empty_cache()
#         gc.collect()
#         pbar.update(1)

#     pbar.close()
#     logger.info(f"GPU {args.gpu_id} has finished all assigned tasks.")

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()

import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import json
import gc
import heapq
import numpy as np
import decord
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import logging 
import torch.nn.functional as F
import warnings

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
# Global Constants (保持不变)
# =========================
TEST_PROMPT_WORLDSENSE = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with **only the letter** (A, B, C, ...) of your choice.

Question: {question}
Options:
{options_str}
Answer:
"""

AKS_MAX_FRAMES = 32
AKS_T1 = 0.8
AKS_T2 = -100
AKS_ALL_DEPTH = 5
MAX_NEW_TOKENS = 512

# =========================
# AKS Algorithm Logic (保持不变)
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
    total_frames_in_video = len(vr)
    
    inputs_text = clip_processor(text=question, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs_text)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        elif not isinstance(text_features, torch.Tensor):
            text_features = text_features[0]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_candidates = min(total_frames_in_video, 96) 
    candidate_indices = np.linspace(0, total_frames_in_video - 1, num_candidates, dtype=int)

    scores, frame_ids = [], []
    for idx in candidate_indices:
        frame = Image.fromarray(vr[idx].asnumpy())
        inputs_img = clip_processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feat = clip_model.get_image_features(**inputs_img)
            if hasattr(img_feat, "pooler_output"):
                img_feat = img_feat.pooler_output
            elif not isinstance(img_feat, torch.Tensor):
                img_feat = img_feat[0]
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        sim = F.cosine_similarity(text_features, img_feat).item()
        scores.append(sim)
        frame_ids.append(int(idx))

    if len(scores) <= AKS_MAX_FRAMES: 
        return sorted(frame_ids)

    score_np = np.array(scores)
    normalized = (score_np - score_np.min()) / (score_np.max() - score_np.min() + 1e-8)
    a, b = meanstd(len(scores), [dict(score=normalized.tolist(), depth=0)], AKS_MAX_FRAMES, [frame_ids], AKS_T1, AKS_T2, AKS_ALL_DEPTH)
    
    selected = []
    for s, f in zip(a, b):
        k = max(1, int(AKS_MAX_FRAMES / (2 ** s["depth"])))
        topk = heapq.nlargest(k, range(len(s["score"])), s["score"].__getitem__)
        selected.extend([f[t] for t in topk])
    
    return sorted(list(set(selected)))[:AKS_MAX_FRAMES]

# =========================
# WorldSense Dataset (修改为串行加载)
# =========================

class WorldSenseDataset(Dataset):
    def __init__(self, json_path, video_root):
        self.video_root = video_root
        with open(json_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)[:5]

        self.samples = []
        # 直接遍历所有数据，不再进行 gpu_id 分片
        for vid in sorted(full_data.keys()):
            item = full_data[vid]
            for task_name, task in item.items():
                if task_name.startswith("task"):
                    self.samples.append({
                        "video_id": vid,
                        "task_name": task_name,
                        "task_data": task,
                        "full_item": item
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        task_name = sample["task_name"]
        task = sample["task_data"]
        
        question = task["question"]
        candidates = task["candidates"]
        alphas = [chr(65 + i) + ". " for i in range(len(candidates))]
        options_str = "\n".join([a + c for a, c in zip(alphas, candidates)])

        prompt = TEST_PROMPT_WORLDSENSE.format(
            question=question,
            options_str=options_str
        )

        v_path = os.path.join(self.video_root, f"{video_id}.mp4")
        
        return {
            "prompt": prompt,
            "v_path": v_path,
            "meta": {
                "video_id": video_id,
                "task_name": task_name,
                "question": question,
                "candidates": candidates,
                "gt_answer": task["answer"],
                "domain": sample["full_item"].get("domain", "unknown"),
                "sub_category": sample["full_item"].get("sub_category", "unknown"),
            }
        }

def collate_fn(batch):
    return batch

# =========================
# Main Script
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--worldsense_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    # 1. 环境与日志初始化
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_dir, "eval.log"), encoding='utf-8'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # 2. 加载 CLIP 模型 (固定在 cuda:3)
    logger.info("Loading CLIP model on cuda:3...")
    clip_path = "/mnt/data2/yangmrl/project/video2text/clip-vit-base-patch32"
    clip_device = torch.device("cuda:3")
    clip_model = CLIPModel.from_pretrained(clip_path).to(clip_device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_path)

    # 3. 加载 Qwen-Omni 模型 
    logger.info(f"Loading Qwen-Omni from {args.model_path} with device_map='auto'")
    if "qwen2.5_omni" in args.model_path.lower():
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    elif "qwen3_omni" in args.model_path.lower():
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model_path, dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    model.eval()

    # 4. 数据加载 (串行)
    dataset = WorldSenseDataset(args.worldsense_json, args.video_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # --- 视频缓存变量 (保持不变) ---
    last_video_id = None
    cached_selected_indices = None
    cached_images_b64 = []  
    # ------------------

    pbar = tqdm(total=len(dataset), desc="Processing WorldSense")

    # 5. 推理循环 (保持逻辑不变)
    for batch in dataloader:
        data = batch[0]
        meta = data["meta"]
        v_id, t_name = meta["video_id"], meta["task_name"]
        save_path = os.path.join(args.output_dir, f"{v_id}_{t_name}.json")

        if os.path.exists(save_path):
            pbar.update(1)
            continue

        try:
            # --- Step A: AKS 选帧与缓存逻辑 (保持不变) ---
            if v_id == last_video_id:
                selected_indices = cached_selected_indices
                images_b64 = cached_images_b64
            else:
                # 使用固定的 clip_device
                selected_indices = aks_select_frames(data["v_path"], meta["question"], clip_model, clip_processor, clip_device)
                vr = decord.VideoReader(data["v_path"], ctx=decord.cpu(0))
                
                images_b64 = []
                for idx in selected_indices:
                    img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
                    buf = BytesIO()
                    img.save(buf, format="JPEG")
                    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                    images_b64.append(b64_str)
                
                last_video_id = v_id
                cached_selected_indices = selected_indices
                cached_images_b64 = images_b64
            # --------------------------------

            # --- Step B: 构造多模态消息 (保持不变) ---
            content = [{"type": "text", "text": data["prompt"]}]
            content.append({"type": "audio", "audio": data["v_path"]})
            for b64_data in images_b64:
                content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64_data}"})
            
            new_message = [{"role": "user", "content": content}]
            text = processor.apply_chat_template([new_message], tokenize=False, add_generation_prompt=True)
            
            audios, images, videos = process_mm_info(new_message, use_audio_in_video=False)

            inputs = processor(
                text=text, audio=audios, images=images, videos=None,
                return_tensors="pt", padding=False, use_audio_in_video=False
            )
            
            
            inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=model.dtype)

            # --- Step C: 模型生成 (保持不变) ---
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    use_audio_in_video=False
                )
                
                if isinstance(generated_ids, tuple):
                    generated_ids = generated_ids[0]
                
                input_len = inputs["input_ids"].shape[1]
                response = processor.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

            # --- Step D: 保存结果 (保持不变) ---
            result = {
                "video_id": v_id,
                "task_name": t_name,
                "question": meta["question"],
                "candidates": meta["candidates"],
                "gt_answer": meta["gt_answer"],
                "prediction": response,
                "raw_response": response,
                "domain": meta["domain"],
                "sub_category": meta["sub_category"],
                "aks_selected_indices": selected_indices 
            }
            with open(save_path, "w", encoding="utf-8") as fw:
                json.dump(result, fw, ensure_ascii=False, indent=2)

            logger.info(f"DONE: {v_id} | {t_name} | Pred: {response} | GT: {meta['gt_answer']}")

        except Exception as e:
            logger.error(f"FAILED: {v_id} | {t_name} | Error: {str(e)}")

        # 深度清理显存 (保持不变)
        del content, new_message, inputs
        if 'generated_ids' in locals(): del generated_ids
        torch.cuda.empty_cache()
        gc.collect()
        pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
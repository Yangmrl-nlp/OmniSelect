import torch
import json
import os
import argparse
import gc
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
MAX_NEW_TOKENS = 512


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

def main():
    parser = argparse.ArgumentParser(description="inference on WorldSense benchmark")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument('--worldsense_json', type=str, required=True, help="WorldSense JSON path")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nframes', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default="./worldsense_results")

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
    TOTAL_PIXELS = NFRAMES * 768 * 28 * 28
    
    cnt = 0
    
    for batch_idx, (messages_batch, indices, metas_batch) in enumerate(tqdm(dataloader, desc="Greedy Inference", total=len(dataloader))):
        for message, original_idx, meta in zip(messages_batch, indices, metas_batch):
            cnt+=1
            
            video_id = meta["video_id"]
            task_name = meta["task_name"]
            
            # if video_id != 'XoquFTEn' or task_name != 'task0':
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
            
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=False,
                use_audio_in_video = True
            )
            
            inputs = inputs.to(model.device).to(model.dtype)
            
            actual_model.eval()
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        thinker_max_new_tokens = 2, 
                        use_audio_in_video=True, 
                        return_audio=False, 
                        temperature=1)
            
            torch.cuda.synchronize()
            peak_mem = 0
            for i in range(torch.cuda.device_count()):
                peak_mem += torch.cuda.max_memory_allocated(i)
                
            peak_mem = peak_mem / 1024**3  
            print(peak_mem)
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

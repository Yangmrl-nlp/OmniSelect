import torch
import json
import os
import re
import sys
import gc
import argparse
import warnings
from tqdm import tqdm
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
TEST_PROMPT_OMNIVIDEO = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with **only the letter** (A, B, C, ...) of your choice.

Question: {question}
Options:
{options_str}
Answer:
"""

MIN_PIXELS = 128 * 28 * 28  
MAX_PIXELS = 128 * 28 * 28 
MAX_NEW_TOKENS = 512

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.vision_process import process_vision_info
from utils.audio_process import process_audio_info
import utils.vision_process


print(f"Vision process script location: {os.path.abspath(utils.vision_process.__file__)}")

def convert_duration_to_seconds(time_str):
    if not time_str or ':' not in str(time_str): return 0
    parts = str(time_str).split(':')
    try:
        if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except: return 0
    return 0

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

def main():
    parser = argparse.ArgumentParser(description="Inference on OmniVideoBench")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--omnivideo_json', type=str, required=True, help="Path to OmniVideoBench data.json")
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of OmniVideoBench")
    parser.add_argument('--nframes', type=int, default=32, help="Number of frames to sample")
    parser.add_argument('--output_dir', type=str, default=os.path.join(current_dir, "results"))
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

    for video_path, prompt_text, meta in tqdm(dataloader, desc="Greedy Inference"):
        if not os.path.exists(video_path):
            continue

        video_id = meta["video_id"]
        sample_idx = meta["idx"]

        save_filename = f"{video_id}_q{sample_idx}.json"
        save_path = os.path.join(output_dir, save_filename)
        cnt+=1
        if cnt!=964:
            continue
        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    existing_result = json.load(f)
                    all_results_list.append(existing_result)
                continue 
            except Exception:
                pass 

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

            text = processor.apply_chat_template(new_message, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_audio_info(new_message, True), *process_vision_info(new_message)

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
         
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    thinker_max_new_tokens=2, 
                    use_audio_in_video=True, 
                    return_audio=False, 
                    temperature=1
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

        except Exception as e:
            print(f"Error on {video_id}: {e}")

        if 'inputs' in locals(): del inputs
        if 'generated_ids' in locals(): del generated_ids
        torch.cuda.empty_cache()
        gc.collect()

    summary_path = os.path.join(output_dir, "_all_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results_list, f, ensure_ascii=False, indent=2)

    total = len(all_results_list)
    correct = sum(1 for r in all_results_list if r['is_correct'])
    print(f"\nDone. Results saved in {output_dir}")
    print(f"Total Accuracy: {correct/total:.2%} ({correct}/{total})")

if __name__ == "__main__":
    main()
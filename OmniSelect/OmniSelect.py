import torch
import numpy as np
import torch.nn as nn
import math
from collections import defaultdict 
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

@torch.no_grad()
def allocate_pruning_weights(scores, num, total_pruning_ratio):
    device = scores.device
    num_tensor = torch.as_tensor(num, device=device, dtype=torch.float32)

    total_tokens = num_tensor.sum()
    total_expected_prune = total_tokens * total_pruning_ratio
    
    scores_std = scores.std()
    if scores_std < 1e-8:
        base_prune_probs = torch.full_like(scores, total_pruning_ratio)
    else:
        norm_scores = (scores - scores.mean()) / (scores_std + 1e-8)
        temperature = 1.0
        base_prune_probs = torch.sigmoid(-norm_scores / temperature)

    current_prune_sum = (base_prune_probs * num_tensor).sum()
    scale_factor = total_expected_prune / (current_prune_sum + 1e-8)
    block_prune_ratios = torch.clamp(base_prune_probs * scale_factor, 0.0, 1.0)

    for _ in range(5):
        current_sum = (block_prune_ratios * num_tensor).sum()
        diff = total_expected_prune - current_sum
        
        if abs(diff) < 1e-3:
            break
    
        adjustable_mask = (block_prune_ratios > 0) & (block_prune_ratios < 1)
        
        if not adjustable_mask.any() or num_tensor[adjustable_mask].sum() < 1e-6:
            break
            
        adjustment = diff / num_tensor[adjustable_mask].sum()
        block_prune_ratios[adjustable_mask] += adjustment
        block_prune_ratios = torch.clamp(block_prune_ratios, 0.0, 1.0)

    return block_prune_ratios

def visualize_omniselect_grid(visual, global_mask, v_lst, video_grid_thw, save_path):
   
    video_mask = global_mask[v_lst].cpu().numpy()
    total_video_tokens = len(video_mask)
    
    _, p_h, p_w = video_grid_thw[0].tolist()
    p_h = p_h // 2
    p_w = p_w // 2
    tokens_per_feat_frame = (p_h * p_w) 
    
    actual_T_feat = total_video_tokens // (tokens_per_feat_frame)
    if isinstance(visual, torch.Tensor):
        visual_np = visual.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
    else:
        visual_np = visual
    
    T_visual = visual_np.shape[0] 
    temporal_merge = T_visual // actual_T_feat 
    
    patch_pixel_h = visual_np.shape[1] / p_h
    patch_pixel_w = visual_np.shape[2] / p_w

    processed_frames = []

    for t in range(0,actual_T_feat):
        start_idx = t * tokens_per_feat_frame
        end_idx = (t + 1) * tokens_per_feat_frame
    
        current_step_mask = video_mask[start_idx:end_idx].reshape(p_h, p_w)
        
        for m in range(temporal_merge):
            frame_idx = t * temporal_merge + m
            if frame_idx >= T_visual: break
            
            img = Image.fromarray(visual_np[frame_idx]).convert("RGBA")
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            for r in range(p_h):
                for c in range(p_w):
                    if not current_step_mask[r, c]:
                        y1, x1 = r * patch_pixel_h, c * patch_pixel_w
                        y2, x2 = (r + 1) * patch_pixel_h, (c + 1) * patch_pixel_w
                        draw.rectangle([x1, y1, x2, y2], fill=(40, 40, 40, 180))
            
            if m == 1:
                processed_frames.append(Image.alpha_composite(img, overlay).convert("RGB"))

    if not processed_frames: return


    sp = save_path.replace('_grid.png', '_single.png')
    target_idx = -9
    processed_frames[target_idx].save(sp)
    print(f"Saved key frame to {sp}")
    cols = 4 
    rows = int(np.ceil(len(processed_frames) / cols))
    full_canvas = Image.new('RGB', (cols * visual_np.shape[2], rows * visual_np.shape[1]), (255, 255, 255))

    for idx, frame in enumerate(processed_frames):
        full_canvas.paste(frame, ((idx % cols) * visual_np.shape[2], (idx // cols) * visual_np.shape[1]))

    full_canvas.save(save_path)
    print(f"Successfully saved. Visual frames: {len(processed_frames)}, Feature frames: {actual_T_feat}")

@torch.no_grad()
def guide_modal_prune(tmp_list,attn_mean,prune_ratio,global_mask,modality):
    device = attn_mean.device
    num_token = len(tmp_list)
    keep_num = int(max(1, num_token * (1.0 - prune_ratio)))
    
    if modality == 'video':
         _, topk_keep_indices = torch.topk(-attn_mean, keep_num, largest=True, sorted=False)
    else:
        _, topk_keep_indices = torch.topk(attn_mean, keep_num, largest=True, sorted=False)
    
    keep_mask = torch.zeros(num_token, dtype=torch.bool, device=device)
    keep_mask[topk_keep_indices] = True
    global_mask[tmp_list] = keep_mask
    
    return global_mask


@torch.no_grad()
def omniselect(
    video_grid_thw,
    input_embeds,
    audio_features,
    vision_features,
    attn_mean_a,
    attn_mean_v,
    input_ids,
    prune_need
):
    device = input_embeds.device
    is_batched = input_embeds.dim() == 3
    if is_batched:
        B, L, D = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, D)
        flat_ids = input_ids.reshape(-1)
    else:
        L, D = input_embeds.shape
        flat_embeds = input_embeds
        flat_ids = input_ids
    
    merge_ratio_a = prune_need['args'].prune_ratio_a
    merge_ratio_v = prune_need['args'].prune_ratio_v
    
    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
    token_ids2group = prune_need['token_id2group']
    logits_a = prune_need['logits_a']
    logits_v = prune_need['logits_v']
    v_lst, a_lst = prune_need['v_lst'], prune_need['a_lst'] 
    cnt = 0
    num_v = []
    num_a = []
    mp = defaultdict(str)
    
    for i in range(len(v_lst)-1):
        cnt += 1
        mp[v_lst[i]] = 'v'
        mp[v_lst[i+1]] = 'v'
        if v_lst[i+1] - v_lst[i] != 1:
            num_v.append(cnt)
            cnt = 0
    num_v.append(cnt+1)
    cnt = 0
    
    for i in range(len(a_lst)-1):
        cnt += 1
        mp[a_lst[i]] = 'a'
        mp[a_lst[i+1]] = 'a'
        if a_lst[i+1] - a_lst[i] != 1:
            num_a.append(cnt)
            cnt = 0
    num_a.append(cnt+1)
    
    if prune_need['video_first'] == True: 
        if len(logits_v) % 4 != 0:
            k = len(logits_v) // 4
            main = logits_v[:k * 4].view(-1, 4)   # [k, 4]
            main_avg = main.mean(dim=1)  
            tail_avg = logits_v[k * 4:].mean().unsqueeze(0)
            pooled_v = torch.cat([main_avg, tail_avg])
        else:
            if len(logits_v) // 4 == len(num_v):
                pooled_v = torch.mean(logits_v.view(-1, 4),dim = 1)
            else:
                k = len(num_v) - 1
                main = logits_v[:k * 4].view(-1, 4) 
                main_avg = main.mean(dim=1)  
                tail_avg = logits_v[k * 4:].mean().unsqueeze(0)
                pooled_v = torch.cat([main_avg, tail_avg])
        
        weights = allocate_pruning_weights(pooled_v,num_v,total_pruning_ratio=merge_ratio_v)
        print("video first, pruning ratio are: ",weights)
        
        cur_timegroup = None
        cur_token_l_v = 0
        cur_token_l_a = 0
        tmp_audio = []
        tmp_video = []
        
        for i in range(len(input_ids)): 
            if mp[i] == 'v':
                tmp_video.append(i)
            elif mp[i] == 'a':
                tmp_audio.append(i)
            else:
                continue

            if cur_timegroup == None:
                cur_timegroup = token_ids2group[i]
            else:
                if token_ids2group[i] != cur_timegroup:
                    attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
                    attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
                    attn_tmp_v = attn_tmp_v.mean(dim=0)
                    global_mask = guide_modal_prune( 
                        tmp_video[:-1], 
                        attn_tmp_v,
                        weights[cur_timegroup - 1],
                        global_mask,
                        'video') 
                    
                    global_mask = guide_modal_prune(
                        tmp_audio,
                        attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                        merge_ratio_a,
                        global_mask,
                        'audio')        
                    
                    cur_token_l_v += len(tmp_video) - 1
                    cur_token_l_a += len(tmp_audio)
                    tmp_audio = [] 
                    tmp_video = tmp_video[-1:]
                    cur_timegroup = token_ids2group[i]
        
        if len(tmp_video) > 0:
            attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
            attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
            attn_tmp_v = attn_tmp_v.mean(dim=0)
            global_mask = guide_modal_prune( 
                tmp_video, 
                attn_tmp_v, 
                weights[cur_timegroup - 1],
                global_mask,
                'video')
        
        if len(tmp_audio) > 0:
            global_mask = guide_modal_prune(
                tmp_audio,
                attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                merge_ratio_a,
                global_mask,
                'audio')
        
    elif prune_need['video_first'] == False: 
        if len(logits_a) > len(logits_v):
            prefix = logits_a[:-2]
            last_avg = logits_a[-2:].mean(dim=0, keepdim=True)
            logits_a = torch.cat([prefix, last_avg], dim=0)
        elif len(logits_a) % 4 != 0:
            k = len(logits_a) // 4
            r = len(logits_a) % 4
            main = logits_a[:k * 4].view(-1, 4)   # [4, k]
            main_avg = main.mean(dim=1)  
            tail_avg = logits_a[k * 4:].mean().unsqueeze(0)
            pooled_a = torch.cat([main_avg, tail_avg])
        else:
            if len(logits_a) // 4 == len(num_a):
                pooled_a = torch.mean(logits_a.view(-1, 4),dim = 1)
            else:
                k = len(num_v) - 1
                main = logits_a[:k * 4].view(-1, 4)   # [k, 4]
                main_avg = main.mean(dim=1)  
                tail_avg = logits_a[k * 4:].mean().unsqueeze(0)
                if len(num_a) > k:
                   pooled_a = torch.cat([main_avg, tail_avg])
                else:
                   main_avg[-1] = (main_avg[-1] + tail_avg[-1]) / 2.0
                   pooled_a = main_avg
                
        
        weights = allocate_pruning_weights(pooled_a,num_a,total_pruning_ratio=merge_ratio_a) 
        print('audio first, pruning ratio are:',weights)
        cur_timegroup = None
        tmp_audio = []
        tmp_video = []
        cur_token_l_v = 0
        cur_token_l_a = 0
        
        for i in range(len(input_ids)): 
            if mp[i] == 'a':
                tmp_audio.append(i)
            elif mp[i] == 'v':
                tmp_video.append(i)
            else:
                continue
            if cur_timegroup == None:
                cur_timegroup = token_ids2group[i]
            else:
                if token_ids2group[i] != cur_timegroup:
                    global_mask = guide_modal_prune(
                        tmp_audio,
                        attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                        weights[cur_timegroup - 1],
                        global_mask,
                        'audio') 
                    
                    attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
                    attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
                    attn_tmp_v = attn_tmp_v.mean(dim=0)
                    global_mask = guide_modal_prune(
                        tmp_video[:-1],
                        attn_tmp_v,
                        merge_ratio_v,
                        global_mask,
                        'video')
                    cur_token_l_v += len(tmp_video) - 1
                    cur_token_l_a += len(tmp_audio)           
                    tmp_audio = [] 
                    tmp_video = tmp_video[-1:]
                    cur_timegroup = token_ids2group[i]
        
        if len(tmp_audio) > 0:
            global_mask = guide_modal_prune(
                tmp_audio,
                attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                weights[cur_timegroup - 1],
                global_mask,
                'audio') 
        
        if len(tmp_video) > 0:
            attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
            attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
            attn_tmp_v = attn_tmp_v.mean(dim=0)
            global_mask = guide_modal_prune(
                tmp_video,
                attn_tmp_v,
                merge_ratio_v,
                global_mask,
                'video')          
    else:
        print("uniform pruning...")
        weights = [merge_ratio_v] * len(num_v) 
        cur_timegroup = None
        cur_token_l_v = 0
        cur_token_l_a = 0
        tmp_audio = []
        tmp_video = []
        
        for i in range(len(input_ids)): 
            if mp[i] == 'a':
                tmp_audio.append(i)
            elif mp[i] == 'v':
                tmp_video.append(i)
            else:
                continue

            if cur_timegroup == None:
                cur_timegroup = token_ids2group[i]
            else:
                if token_ids2group[i] != cur_timegroup:
                    attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
                    attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
                    attn_tmp_v = attn_tmp_v.mean(dim=0)
                    global_mask = guide_modal_prune( 
                        tmp_video[:-1], 
                        attn_tmp_v,
                        merge_ratio_v,
                        global_mask,
                        'video') 
                    
                    global_mask = guide_modal_prune(
                        tmp_audio,
                        attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                        merge_ratio_a,
                        global_mask,
                        'audio')        
                   
                    cur_token_l_v += len(tmp_video) - 1
                    cur_token_l_a += len(tmp_audio)
                    tmp_audio = [] 
                    tmp_video = tmp_video[-1:]
                    cur_timegroup = token_ids2group[i]
        
        if len(tmp_video) > 0:
            attn_tmp_v = attn_mean_v[cur_token_l_v : cur_token_l_v + len(tmp_video) - 1]
            attn_tmp_v = attn_tmp_v @ attn_tmp_v.T
            attn_tmp_v = attn_tmp_v.mean(dim=0)
            global_mask = guide_modal_prune( 
                tmp_video, 
                attn_tmp_v, 
                merge_ratio_v,
                global_mask,
                'video')
        if len(tmp_audio) > 0:
            global_mask = guide_modal_prune(
                tmp_audio,
                attn_mean_a[cur_token_l_a : cur_token_l_a + len(tmp_audio) - 1],
                merge_ratio_a,
                global_mask,
                'audio')
            
    return global_mask

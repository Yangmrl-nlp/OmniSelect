import os
import json
import glob
import re
from pathlib import Path
from collections import defaultdict 

RESULTS_DIR = Path("/mnt/data2/yangmrl/project/video2text/eval/dailyomni_eval/results/frame_128/qwen2.5_omni_7b_ours_45_100")
RESULTS_DIR_ = Path("/mnt/data2/yangmrl/project/video2text/eval/dailyomni_eval/results/frame_128/qwen2.5_omni_7b_ours_45_-100")


def extract_answer(text):
    if not text or len(text) > 2: 
        return ""
    else:
        return text[0]

def main():
    mp = defaultdict(int)
    su = defaultdict(int)
    mp_ = defaultdict(int)
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    result_files_ = glob.glob(os.path.join(RESULTS_DIR_, "*.json"))
    for file_path in result_files_:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw_pred = data.get("prediction", "")
            gt = data.get("gt_letter", "").strip().upper()
            pred = extract_answer(raw_pred)
            if pred == gt:
                mp_[data['video_id']+data['q_id']] += 1

    total = 0
    correct = 0
    cnt = 0
    cnt_ = 0
    for file_path in result_files:
        cnt+=1
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw_pred = data.get("prediction", "")
            gt = data.get("gt_letter", "").strip().upper()
            pred = extract_answer(raw_pred)
            su[data['domain']]+=1
            # if data['pre_abs'][0] > data['pre_abs'][1]:
            #     cnt_+=1
            if pred == gt or mp_[data['video_id']+data['q_id']]:
                mp[data['domain']] += 1
                correct += 1
            # elif mp_[data['video_id']+data['task_name']]:
            #     print(data['video_id'],data['task_name'])
            total += 1
    if total == 0:
        print("check path.")
        return

    accuracy = (correct / total) * 100
    print(f"sum of data: {total}")
    print(f"correct: {correct}")
    print(f"Avg. is: {accuracy:.2f}%")
    print('*'*60)
    pre = 0
    cnt = 0
    items = list(mp.items())
    items[-1], items[-2] = items[-2], items[-1]
    mp = dict(items)
    
    items_ = list(su.items())
    items_[-1], items_[-2] = items_[-2], items_[-1]
    su = dict(items_)
    for t in su.keys():
        print(f'{t} accuraccy: {((mp[t] / su[t])*100):.2f}')
        pre += ((mp[t] / su[t])*100)
        cnt+=1
    print('*'*60)
        
    
if __name__ == "__main__":
    main()

# temp = []
# for t in RESULTS_DIR.iterdir():
#     if t.stem == '_all_results_summary':
#         continue
#     with open(t,'r') as f:
#         data = json.load(f)
    
#     raw_pred = data.get("prediction", "")
#     gt = data.get("gt_answer", "").strip().upper()
#     pred = extract_answer(raw_pred)
#     if pred != gt:
#         temp.append(data)
    
# with open('/mnt/data2/yangmrl/project/video2text/eval/Worldsense_eval/results/case_study/temp.json','w') as f:
#     json.dump(temp,f,indent=4,ensure_ascii=False)


# data = []
# with open(RESULTS_DIR,'r') as f:
#     for line in f :
#        data.append(json.loads(line))

# cnt = 0
# sum = 0
# for dat in data:
    
#     raw_pred = dat.get("prediction", "")
#     gt = dat.get("gt_answer", "").strip().upper()
#     if sum == 2512:
#         break
#     pred = extract_answer(raw_pred)
#     sum += 1
#     # if sum == 999:
#     #     break
#     if pred == gt:
#         cnt+=1
# print(cnt,sum)

# data1 = []
# data2 = []
# with open(RESULTS_DIR,'r') as f:
#      data1 = json.load(f)
# with open(RESULTS_DIR_,'r') as f:
#      data2 = json.load(f)
# cnt = 0
# cnt_u = 0
# cnt_s = 0
# cor = 0
# for i in range(len(data1)):
#     dat = data1[i]
#     dat_ = data2[i]
#     raw_pred = dat.get("prediction", "")
#     pred = extract_answer(raw_pred)
#     gt = dat.get("gt_answer", "").strip().upper()
    
#     raw_pred_ = dat_.get("prediction", "")
#     pred_ = extract_answer(raw_pred_)
#     gt_ = dat_.get("gt_answer", "").strip().upper()
#     if pred == gt and pred_ == gt_:
#         cnt+=1
#         continue
#     elif pred != gt and pred_ != gt_:
#         continue
#     if pred == gt:
#         # print(dat['pre_abs'],1,dat['video_id'])
#         cha = abs(dat['pre_abs'][0]-dat['pre_abs'][1])
#         if cha <=3.5:
#             cnt+=1
#         if cha > 1 and cha<=3:
#             cor+=0
#     else:
#         print(dat['pre_abs'],0,dat['video_id'])
#         cha = abs(dat['pre_abs'][0]-dat['pre_abs'][1])
#         if cha >3.5:
#             cnt+=1
        
#         if cha > 1 and cha<=3:
#             cor+=1
        
# print(cor,cnt)

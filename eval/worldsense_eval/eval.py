import os
import json
import glob
import re
from pathlib import Path
from collections import defaultdict 

RESULTS_DIR = Path("path/to/your/worldsense_results")


def extract_answer(text):
    if not text or len(text) > 2: 
        return ""
    else:
        return text[0]

def main():
    mp = defaultdict(int)
    su = defaultdict(int)
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))

    total = 0
    correct = 0
    cnt = 0
    for file_path in result_files:
        cnt+=1
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw_pred = data.get("prediction", "")
            gt = data.get("gt_answer", "").strip().upper()
            pred = extract_answer(raw_pred)
            su[data['domain']]+=1
           
            if pred == gt:
                mp[data['domain']] += 1
                correct += 1
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

import json
import os
import glob
from collections import defaultdict

def evaluate_accuracy():

    gt_path = "/path/to/dailyomni/qa.json"
    results_dir = "/path/to/result.json"

    mp = defaultdict(int)
    su = defaultdict(int)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)

    pred_files = glob.glob(os.path.join(results_dir, "*.json"))
    correct_count = 0
    total_count = 0
    errors = 0
    for file_path in pred_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
            
            qid = pred_data.get("qid")
            prediction = pred_data.get("prediction", "").strip().upper()[0]
            category = gt_list[qid].get("Type")

            if qid is not None and 0 <= int(qid) < len(gt_list):
                gt_item = gt_list[int(qid)]
                gt_answer = gt_item.get("Answer", "").strip().upper()

                su[category] += 1
                if prediction == gt_answer:
                    mp[category] += 1
                    correct_count += 1
                
                total_count += 1
            else:
                print(f"warning: file {os.path.basename(file_path)} qid {qid} exceed Ground Truth length or invalid.")
                errors += 1

        except Exception as e:
            print(f"processing {file_path} error: {e}")
            errors += 1

    count = 0
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("-" * 30)
        print(f"sum of data: {total_count}")
        print(f"correct: {correct_count}")
        print(f"erros: {errors}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("-" * 30)
        for t in su.keys():
            count += su[t]
            print(f'{t} accuraccy: {((mp[t] / su[t])*100):.2f}%')
        print(count)
        print("-" * 30)
    else:
        print("no item")

if __name__ == "__main__":
    evaluate_accuracy()
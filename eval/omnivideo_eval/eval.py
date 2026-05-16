import json
from collections import defaultdict
from pathlib import Path

result_path = Path("/path/to/your/results_dir")
json_path = "/path/to/your/json_file.json"

dat = []
for t in result_path.iterdir():
    if t.stem == '_all_results_summary':
        continue
    with open(t,'r') as f:
        tem = json.load(f)
    dat.append(tem)

with open(json_path,'w') as f:
    json.dump(dat,f,ensure_ascii=False, indent=2)

def evaluate():
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return

    total = len(data)
    if total == 0:
        return

    correct = 0
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in data:
        is_correct = item.get("is_correct", False)
        q_type = item.get("question_type", "Unknown")
       
        if is_correct:
            correct += 1
        
        type_stats[q_type]["total"] += 1
        if is_correct:
            type_stats[q_type]["correct"] += 1

    print("\n" + "="*50)
    print("="*50)
    
    print(f"{'Question Type':<30} | {'Accuracy':<10} | {'Count'}")
    print("-" * 50)
    

    for q_type, stats in sorted(type_stats.items()):
        acc = stats["correct"] / stats["total"]
        print(f"{q_type:<30} | {acc:>8.2%} | {stats['correct']}/{stats['total']}")

    print("-" * 50)
    total_acc = correct / total
    print(f"{'TOTAL ACCURACY':<30} | {total_acc:>8.2%} | {correct}/{total}")
    print("="*50 + "\n")


if __name__ == "__main__":
    evaluate()

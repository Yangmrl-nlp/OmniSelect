import os
import json
import glob

RESULTS_DIR = "/mnt/data2/yangmrl/project/video2text/Worldsense_eval/results/qwen2.5_omni_3b_aks/"

def extract_answer(text):
    if not text: return ""
    return str(text).strip().upper()

def main():
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    
    total = 0
    correct = 0

    for file_path in result_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            raw_pred = data.get("prediction", "")
            gt = data.get("gt_answer", "").strip().upper()
            
            pred = extract_answer(raw_pred)
            
            if pred == gt:
                correct += 1
            total += 1

    if total == 0:
        print("没有找到结果文件，请检查路径。")
        return

    accuracy = (correct / total) * 100
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率 (Accuracy): {accuracy:.2f}%")

if __name__ == "__main__":
    main()
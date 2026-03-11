import json
import os
import glob

def evaluate_accuracy():
    # 路径配置
    gt_path = "/mnt/data2/yangmrl/project/video2text/test_data/dailyomni/qa.json"
    results_dir = "/mnt/data2/yangmrl/project/video2text/dailyomni_eval/results/qwen2.5_omni_7b/"

    # 1. 加载 Ground Truth 数据
    if not os.path.exists(gt_path):
        print(f"错误: 找不到 Ground Truth 文件 {gt_path}")
        return

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)

    # 2. 获取所有的预测结果文件
    pred_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not pred_files:
        print(f"错误: 在目录 {results_dir} 中未找到任何 json 结果文件")
        return

    correct_count = 0
    total_count = 0
    errors = 0

    print(f"开始评估，共发现 {len(pred_files)} 个预测结果文件...")

    # 3. 遍历并比对
    for file_path in pred_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
            
            # 获取结果中的 qid 和预测选项
            qid = pred_data.get("qid")
            prediction = pred_data.get("prediction", "").strip().upper()

            # 检查 qid 是否合法 (作为下标)
            if qid is not None and 0 <= int(qid) < len(gt_list):
                gt_item = gt_list[int(qid)]
                gt_answer = gt_item.get("Answer", "").strip().upper()

                # 比对
                if prediction == gt_answer:
                    correct_count += 1
                
                total_count += 1
            else:
                print(f"警告: 文件 {os.path.basename(file_path)} 的 qid {qid} 超出 Ground Truth 列表范围或无效。")
                errors += 1

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            errors += 1

    # 4. 输出最终结果
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("-" * 30)
        print(f"评估完成:")
        print(f"总样本数 (有效): {total_count}")
        print(f"正确数量: {correct_count}")
        print(f"解析失败/范围错误: {errors}")
        print(f"准确率 (Accuracy): {accuracy:.2f}%")
        print("-" * 30)
    else:
        print("没有可统计的有效样本。")

if __name__ == "__main__":
    evaluate_accuracy()
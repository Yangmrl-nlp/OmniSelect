export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /mnt/data2/yangmrl/project/video2text/AVQA_eval


# for ((i=0; i<=1033; i++)); do
python infer_avqa.py \
    --model_path "/mnt/data2/yangmrl/project/video2text/models/qwen2.5_omni_3b" \
    --avqa_json "/mnt/data2/yangmrl/project/video2text/test_data/AVQA/AVQA_dataset/val_qa.json" \
    --output_dir "./results/qwen2.5_omni_3b" \
    --batch_size 1 \
    # --qid $i
# done
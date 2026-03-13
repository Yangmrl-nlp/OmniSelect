export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/mnt/data2/yangmrl/project/video2text:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
cd /mnt/data2/yangmrl/project/video2text/dailyomni_eval

python infer_dailyomni.py \
    --model_path "/mnt/data2/yangmrl/project/video2text/models/qwen2.5_omni_7b" \
    --daily_json "/mnt/data2/yangmrl/project/video2text/test_data/dailyomni/qa.json" \
    --video_root "/mnt/data2/yangmrl/project/video2text/test_data/dailyomni/Videos" \
    --output_dir "./results/qwen2.5_omni_7b" \




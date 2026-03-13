export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /mnt/data2/yangmrl/project/video2text/Worldsense_eval


python infer_worldsense_ours.py \
    --model_path "/mnt/data2/yangmrl/project/video2text/models/qwen2.5_omni_3b" \
    --worldsense_json "/mnt/data2/yangmrl/project/video2text/test_data/worldsense/worldsense_qa.json" \
    --output_dir "./results/qwen2.5_omni_3b_ours" \
    # --video_root "/mnt/data2/yangmrl/project/video2text/test_data/worldsense/videos" \
    # --batch_size 1 \
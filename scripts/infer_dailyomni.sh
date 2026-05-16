export CUDA_VISIBLE_DEVICES=1,2,3

cd /path/to/eval/dailyomni_eval

#Full Tokens Inference
python infer_dailyomni.py \
    --model_path "/path/to/models/qwen2.5_omni_3b" \
    --DailyOmni_json "/path/to/test_data/dailyomni/qa.json" \
    --video_root "/path/to/test_data/dailyomni/Videos" \
    --output_dir "./results/qwen2.5_omni_3b" \
    --nframes 128


#OmniSelect Inference

python infer_dailyomni_ours.py \
    --model_path "/path/to/models/qwen2.5_omni_3b" \
    --DailyOmni_json "/path/to/test_data/dailyomni/qa.json" \
    --video_root "/path/to/test_data/dailyomni/Videos" \
    --output_dir "./results/frame_128/qwen2.5_omni_3b_ours_45" \
    --prune_ratio_a 0.55 \
    --prune_ratio_v 0.55 \
    --theta 2\
    --nframes 128



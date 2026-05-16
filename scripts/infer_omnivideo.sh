export CUDA_VISIBLE_DEVICES=1,2,3
cd /path/to/eval/omnivideo_eval

#FULL Tokens
python infer_omnivideo.py \
    --model_path "/path/to/models/qwen2.5_omni_3b" \
    --omnivideo_json "/path/to/test_data/OmniVideoBench/data.json" \
    --data_root "/path/to/test_data/OmniVideoBench/" \
    --output_dir "./results/frame_128/qwen2.5_omni_3b" \
    --nframes 128


#OmniSelect
python infer_omnivideo_ours.py \
    --model_path "/path/to/models/qwen2.5_omni_7b" \
    --omnivideo_json "/path/to/test_data/OmniVideoBench/data.json" \
    --data_root "/path/to/test_data/OmniVideoBench/" \
    --output_dir "./results/frame_128/qwen2.5_omni_7b_ours_30" \
    --prune_ratio_a 0.70 \
    --prune_ratio_v 0.70 \
    --theta 2 \
    --nframes 128


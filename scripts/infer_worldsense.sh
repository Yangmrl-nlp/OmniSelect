export CUDA_VISIBLE_DEVICES=1,2,3
cd /path/to/eval/Worldsense_eval

#Full tokens inference
python infer_worldsense.py \
    --model_path "/path/to/models/qwen2.5_omni_3b" \
    --worldsense_json "/path/to/test_data/worldsense/worldsense_qa.json" \
    --output_dir "./results/frame_128/qwen2.5_omni_3b" \
    --nframes 128

#Omniselect inference

python infer_worldsense_ours.py \
    --model_path "/path/to/models/qwen2.5_omni_3b" \
    --worldsense_json "/path/to/test_data/worldsense/worldsense_qa.json" \
    --output_dir "./results/frame_128/qwen2.5_omni_3b_ours_30" \
    --prune_ratio_a 0.70 \
    --prune_ratio_v 0.70 \
    --theta 2 \
    --nframes 128


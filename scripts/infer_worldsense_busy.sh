#!/bin/bash

export PYTHONPATH=/mnt/data2/yangmrl/project/video2text:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/data2/yangmrl/project/video2text/Worldsense_eval

MODEL_PATH="/mnt/data2/yangmrl/project/video2text/models/qwen2.5_omni_3b"
JSON_PATH="/mnt/data2/yangmrl/project/video2text/test_data/worldsense/worldsense_qa.json"
VIDEO_ROOT="/mnt/data2/yangmrl/project/video2text/test_data/worldsense/videos"
OUTPUT_DIR="./results/qwen2.5_omni_3b_aks"
NUM_GPUS=4

for i in {0..3}
do
    echo "Starting process for GPU $i..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_worldsense_aks.py \
        --model_path "$MODEL_PATH" \
        --worldsense_json "$JSON_PATH" \
        --video_root "$VIDEO_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_id $i \
        --num_gpus $NUM_GPUS & 
done

wait
echo "All Done!"


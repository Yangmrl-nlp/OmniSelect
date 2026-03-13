#!/bin/bash
#
# Single Sample Inference with LongVT
#
# This script starts a vLLM server and runs single sample inference.
# For full benchmark evaluation, use run_eval.sh instead.
#
# Usage:
#   bash run_single_inference.sh <CKPT_PATH> <VIDEO_PATH> <QUESTION> [IS_QWEN3_VL] [DP_SIZE] [GPU_UTIL] [PORT]
#
# Arguments:
#   CKPT_PATH     - Path to model checkpoint (required)
#   VIDEO_PATH    - Path to video file (required)
#   QUESTION      - Question about the video (required)
#   IS_QWEN3_VL   - Whether model is Qwen3-VL (default: False)
#   DP_SIZE       - Data parallel size for multi-GPU (default: 1, use 8 for 8 GPUs)
#   GPU_UTIL      - GPU memory utilization 0.0-1.0 (default: 0.9)
#   PORT          - vLLM server port (default: 8000)
#
# Examples:
#   # Single GPU inference
#   bash run_single_inference.sh \
#       longvideotool/LongVT-7B-RFT \
#       /path/to/video.mp4 \
#       "What is happening in the video?"
#
#   # Multi-GPU inference (8 GPUs with 70% memory utilization)
#   bash run_single_inference.sh \
#       longvideotool/LongVT-7B-RFT \
#       /path/to/video.mp4 \
#       "What is happening in the video?" \
#       False 8 0.7

CKPT_PATH=$1
VIDEO_PATH=$2
QUESTION=$3
IS_QWEN3_VL=${4:-False}
DP_SIZE=${5:-1}
GPU_UTIL=${6:-0.9}
PORT=${7:-8000}

if [ -z "$CKPT_PATH" ] || [ -z "$VIDEO_PATH" ] || [ -z "$QUESTION" ]; then
    echo "Usage: bash run_single_inference.sh <CKPT_PATH> <VIDEO_PATH> <QUESTION> [IS_QWEN3_VL] [DP_SIZE] [GPU_UTIL] [PORT]"
    echo ""
    echo "Arguments:"
    echo "  CKPT_PATH     - Path to model checkpoint (required)"
    echo "  VIDEO_PATH    - Path to video file (required)"
    echo "  QUESTION      - Question about the video (required)"
    echo "  IS_QWEN3_VL   - Whether model is Qwen3-VL (default: False)"
    echo "  DP_SIZE       - Data parallel size for multi-GPU (default: 1)"
    echo "  GPU_UTIL      - GPU memory utilization 0.0-1.0 (default: 0.9)"
    echo "  PORT          - vLLM server port (default: 8000)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bypass proxy for localhost (important for local vLLM server)
export no_proxy=localhost,127.0.0.1

echo "[INFO] Starting vLLM server..."
echo "[INFO] Model: $CKPT_PATH"
echo "[INFO] Data Parallel Size: $DP_SIZE"
echo "[INFO] GPU Memory Utilization: $GPU_UTIL"
echo "[INFO] Port: $PORT"

# Build vLLM command with configurable parameters
VLLM_CMD="vllm serve $CKPT_PATH"
VLLM_CMD+=" --max-model-len 40000" 
VLLM_CMD+=" --tool-call-parser hermes"
VLLM_CMD+=" --enable-auto-tool-choice"
VLLM_CMD+=" --trust-remote-code"
VLLM_CMD+=" --port $PORT"
VLLM_CMD+=" --gpu-memory-utilization $GPU_UTIL"

# Add data parallel size if > 1
if [ "$DP_SIZE" -gt 1 ]; then
    VLLM_CMD+=" --data-parallel-size $DP_SIZE"
fi

# Add chat template for non-Qwen3-VL models
if [ "$IS_QWEN3_VL" != "True" ]; then
    VLLM_CMD+=" --chat-template ${SCRIPT_DIR}/tool_call_qwen2_5_vl.jinja"
fi

echo "[INFO] Command: $VLLM_CMD"
$VLLM_CMD &

VLLM_PID=$!
echo "[INFO] vLLM server PID: $VLLM_PID"

# Wait for vLLM to be ready (poll /v1/models endpoint)
echo "[INFO] Waiting for vLLM server to be ready..."
MAX_WAIT=3600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q '"data"'; then
        echo "[INFO] vLLM server is ready! (took ${WAITED}s)"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "[INFO] Still waiting... ${WAITED}s"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[ERROR] vLLM server failed to start within ${MAX_WAIT}s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run inference
echo "[INFO] Running inference..."
python ${SCRIPT_DIR}/longvt_infer.py \
    --video_path "$VIDEO_PATH" \
    --question "$QUESTION" \
    --api_base "http://localhost:$PORT/v1" \
    --fps 1 \
    --max_frames 512 \
    --max_pixels 50176

# Cleanup
echo "[INFO] Cleaning up..."
kill $VLLM_PID 2>/dev/null
echo "[INFO] Done."

# bash run_single_longvt_inference.sh \
#     /mnt/data2/yangmrl/project/video2text/models/longvt_rl_7B \
#       /mnt/data2/yangmrl/project/video2text/test_data/video_mme/video_data/data/fFjv93ACGo8.mp4 \
#       "please count how many apples,berries,candles from 43 second to 44 second."
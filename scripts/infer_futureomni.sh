export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /mnt/data2/yangmrl/project/video2text/FutureOmni_eval/eval


# for ((i=1033; i<=1033; i++)); do
python infer_ddp.py \
    --model_path "/mnt/data2/yangmrl/project/video2text/models/qwen2.5_omni_3b" \
    --data_file "/mnt/data2/yangmrl/project/video2text/test_data/futureomni/futureomni_test.json" \
    --output_dir "./results" \
    --batch_size 1 \
    --qid 1
# done
#huggingface-cli download --repo-type dataset --resume-download liarliar/Daily-Omni --local-dir /mnt/data2/yangmrl/project/video2text/test_data/dailyomni
#!/bin/bash
set -e

BASE_MODEL="Qwen/Qwen2-7B-Instruct"
OUT_DIR="checkpoints_200"
FUSED_DIR="fused_recall_200"

# ⭐ shell 控制 dataset 用量 (0=全部, 200=debug)
DATA_LIMIT=200

declare -A EPOCHS=( ["sst2"]=3 ["squad2"]=4 ["iwslt2017"]=5 ["race"]=5 ["medmcqa"]=3 )
declare -A BS=( ["sst2"]=64 ["squad2"]=32 ["iwslt2017"]=64 ["race"]=128 ["medmcqa"]=64 )

echo "========== [STEP 0] Train each task =========="

for TASK in sst2 squad2 iwslt2017 race medmcqa; do
  python train_single_task.py \
      --task $TASK \
      --base_model $BASE_MODEL \
      --out_dir $OUT_DIR \
      --subset $DATA_LIMIT \
      --epochs ${EPOCHS[$TASK]} \
      --batch_size ${BS[$TASK]} \
      --lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
      2>&1 | tee logs/train_${TASK}.log
done

echo "✅ 所有 LoRA training 完成"

echo "========== [STEP 1] RECALL merge =========="
python merge_recall.py \
    --base_model $BASE_MODEL \
    --adapters_root $OUT_DIR \
    --out_dir $FUSED_DIR \
    2>&1 | tee logs/merge.log

echo "========== [STEP 2] Evaluate =========="
python evaluate_all_tasks.py \
    --model $FUSED_DIR \
    --tokenizer $BASE_MODEL \
    2>&1 | tee logs/eval.log

echo "🎉 RECALL pipeline 完成"

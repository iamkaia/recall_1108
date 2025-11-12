#!/bin/bash
set -e

# =============================
# 💡 RECALL full pipeline (paper reproduction)
# =============================

BASE_MODEL="Qwen/Qwen2-7B-Instruct"
OUT_DIR="checkpoints_full"
FUSED_DIR="fused_recall_full"
SUBSET=0               # 0 表示不用 subset，全量訓練
EPOCHS=3               # 論文設定
SPLIT_TRAIN="train"
SPLIT_TEST="test"
TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")

LOG_DIR="logs"
mkdir -p $OUT_DIR $LOG_DIR
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M).log"

echo "========== [STEP 0] Environment setup ==========" | tee -a $LOG_FILE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========== [STEP 1] Train each task (subset=$SUBSET, epochs=$EPOCHS) ==========" | tee -a $LOG_FILE
for TASK in "${TASKS[@]}"; do
    CKPT_PATH="$OUT_DIR/$TASK"
    if [ -d "$CKPT_PATH" ] && [ -f "$CKPT_PATH/adapter_config.json" ]; then
        echo "✅ Skip $TASK, already trained at $CKPT_PATH" | tee -a $LOG_FILE
    else
        echo "🚀 Training $TASK ..." | tee -a $LOG_FILE
        python train_single_task.py \
            --base_model $BASE_MODEL \
            --task $TASK \
            --subset $SUBSET \
            --epochs $EPOCHS \
            --out_dir $OUT_DIR 2>&1 | tee -a $LOG_FILE
    fi
done

echo "========== [STEP 2] Convert safetensors → bin if needed ==========" | tee -a $LOG_FILE
python - <<'PYCODE'
import os, torch
from safetensors.torch import load_file

tasks = ["sst2","squad2","iwslt2017","race","medmcqa"]
base_dir = "checkpoints_full"
for t in tasks:
    path = os.path.join(base_dir, t, "adapter_model.safetensors")
    if os.path.exists(path):
        bin_path = os.path.join(base_dir, t, "adapter_model.bin")
        if not os.path.exists(bin_path):
            print(f"⚙️ Converting {path} → {bin_path}")
            state = load_file(path)
            torch.save(state, bin_path)
print("✅ safetensors → bin conversion complete.")
PYCODE

echo "========== [STEP 3] Merge adapters (RECALL fusion) ==========" | tee -a $LOG_FILE
python merge_recall.py \
  --base_model $BASE_MODEL \
  --adapters $OUT_DIR/sst2 $OUT_DIR/squad2 $OUT_DIR/iwslt2017 $OUT_DIR/race $OUT_DIR/medmcqa \
  --weights 0.9 0.8 0.8 0.7 0.7 \
  --out_dir $FUSED_DIR 2>&1 | tee -a $LOG_FILE

echo "========== [STEP 4] Evaluate all tasks (split=$SPLIT_TEST, full dataset) ==========" | tee -a $LOG_FILE
python evaluate_all_tasks.py \
  --model $FUSED_DIR \
  --base_model $BASE_MODEL \
  --split $SPLIT_TEST \
  --max_examples 0 2>&1 | tee -a $LOG_FILE

echo "========== [STEP 5] Analyze results ==========" | tee -a $LOG_FILE
python analyze_results.py | tee -a $LOG_FILE

echo "✅ Finished full RECALL reproduction (paper setting)" | tee -a $LOG_FILE
echo "📂 Check outputs:" | tee -a $LOG_FILE
echo "   - $OUT_DIR/" | tee -a $LOG_FILE
echo "   - $FUSED_DIR/" | tee -a $LOG_FILE
echo "   - $LOG_FILE" | tee -a $LOG_FILE
echo "   - recall_results_full.txt, summary_table.csv" | tee -a $LOG_FILE

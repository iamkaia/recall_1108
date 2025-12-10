#!/usr/bin/env bash
set -euo pipefail

# =============================
# Evaluate Base Model (no training, no merge)
# =============================

BASE_MODEL="unsloth/llama-2-7b-chat"
DATA_ROOT="./datasets"
RESULTS_DIR="results_eval_basemodel_$(date +%Y%m%d_%H%M)"
MAX_LEN_EVAL=2048
MAX_NEW_TOKENS=64

mkdir -p "$RESULTS_DIR"

# 任務清單（論文五個）
TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")

# 每個 task 對應不同抽樣樣本數
# 格式："task:sample_size"
TASK_SAMPLES=(
  "sst2:500"
  "squad2:300"
  "iwslt2017:500"
  "race:1200"
  "medmcqa:1200"
)

echo "========== Evaluate base model with per-task sample sizes =========="

for item in "${TASK_SAMPLES[@]}"; do
    IFS=":" read -r TASK SAMPLE <<< "$item"

    echo "📌 Evaluating: $TASK (samples = $SAMPLE)"

    python evaluate_all_tasks.py \
      --model "$BASE_MODEL" \
      --base_model "$BASE_MODEL" \
      --data_root "$DATA_ROOT" \
      --results_dir "$RESULTS_DIR" \
      --max_examples "$SAMPLE" \
      --task "$TASK" \
      --max_src_len "$MAX_LEN_EVAL" \
      --max_new_tokens "$MAX_NEW_TOKENS"
done

# -------------------------------
# Summary
# -------------------------------
echo "========== [STEP 4] Summarize ==========" | tee -a "$LOG_FILE"
python analyze_results.py 2>&1 | tee -a "$LOG_FILE"

echo "🎉 RECALL 完成" | tee -a "$LOG_FILE"
echo "📁 Outputs:" | tee -a "$LOG_FILE"
echo "   - $OUT_DIR/" | tee -a "$LOG_FILE"
echo "   - $FUSED_DIR/" | tee -a "$LOG_FILE"
echo "   - $RESULTS_DIR/" | tee -a "$LOG_FILE"
echo "   - $LOG_FILE" | tee -a "$LOG_FILE"

#!/usr/bin/env bash
set -euo pipefail

# =============================
# Evaluate Base Model (no training, no merge)
# =============================

BASE_MODEL="unsloth/llama-2-7b-chat"
DATA_ROOT="./datasets_llama"
RESULTS_DIR="results_eval_basemodel_$(date +%Y%m%d_%H%M)"
MAX_LEN_EVAL=2048
MAX_NEW_TOKENS=64
#### HF_TOKEN

mkdir -p "$RESULTS_DIR"

# 依 RECALL 論文 Task 設定不同 sample 數
TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")

TASK_SAMPLES="sst2:500,squad2:300,iwslt2017:500,race:1200,medmcqa:1200"

echo "========== Evaluate all tasks in single run =========="

python evaluate_llama2_official_baseline.py \
  --model "$BASE_MODEL" \
  --data_root "$DATA_ROOT" \
  --results_dir "$RESULTS_DIR" \
  --tasks "${TASKS[@]}" \
  --sample_map "$TASK_SAMPLES" \
  --max_src_len "$MAX_LEN_EVAL" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --hf_token "$HF_TOKEN"

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

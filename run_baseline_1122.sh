#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
DATA_ROOT="./datasets_llama"
RESULTS_DIR="results_eval_basemodel_$(date +%Y%m%d_%H%M)"
MAX_LEN_EVAL=2048
MAX_NEW_TOKENS=64

mkdir -p "$RESULTS_DIR"

TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")
TASK_SAMPLES="sst2:500,squad2:300,iwslt2017:500,race:1200,medmcqa:1200"

echo "========== Evaluate all tasks =========="

python evaluate_all_tasks_llama_v2.py \
  --model "$BASE_MODEL" \
  --data_root "$DATA_ROOT" \
  --results_dir "$RESULTS_DIR" \
  --tasks "${TASKS[@]}" \
  --sample_map "$TASK_SAMPLES" \
  --max_src_len "$MAX_LEN_EVAL" \
  --max_new_tokens "$MAX_NEW_TOKENS"

echo "===== DONE ====="
echo "Results saved to: $RESULTS_DIR"

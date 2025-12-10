#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="/home/kaia/recall_1108/Llama-2-7b-chat-hf"
DATA_ROOT="./datasets_llama"
RESULTS_DIR="results_eval_oc_$(date +%Y%m%d_%H%M)"
MAX_LEN_EVAL=2048
MAX_NEW=64

TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")
TASK_SAMPLES="sst2:500,squad2:300,iwslt2017:500,race:1200,medmcqa:1200"

mkdir -p "$RESULTS_DIR"

echo "========== OpenCompass-style Baseline =========="

python evaluate_llama2_opencompass_baseline.py \
  --model "$BASE_MODEL" \
  --data_root "$DATA_ROOT" \
  --results_dir "$RESULTS_DIR" \
  --tasks "${TASKS[@]}" \
  --sample_map "$TASK_SAMPLES" \
  --max_src_len "$MAX_LEN_EVAL" \
  --max_new_tokens "$MAX_NEW" \
  --debug_k 5

echo "===== DONE ====="
echo "Results saved to: $RESULTS_DIR"

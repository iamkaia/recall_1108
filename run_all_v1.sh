#!/bin/bash
# run_all.sh (subset=500 version for quick RECALL reproduction)

BASE="Qwen/Qwen2-7B-Instruct"
OUT="checkpoints_500"
FUSED="fused_recall_500"
SUBSET=200

TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")
WEIGHTS=(0.9 0.8 0.8 0.7 0.7)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========== [STEP 1] Train each task (subset=${SUBSET}) =========="

for t in "${TASKS[@]}"; do
  SAVE_DIR="${OUT}/${t}"
  if [ -d "$SAVE_DIR" ]; then
    echo "✅ Skip ${t}, already trained at ${SAVE_DIR}"
  else
    echo "🚀 Training ${t} ..."
    python train_single_task.py \
      --base_model $BASE \
      --task $t \
      --subset $SUBSET \
      --out_dir $OUT
  fi
done

echo "========== [STEP 2] Merge adapters (RECALL fusion) =========="

ADAPTERS=()
for t in "${TASKS[@]}"; do
  ADAPTERS+=("${OUT}/${t}")
done

python merge_recall.py \
  --base_model $BASE \
  --adapters "${ADAPTERS[@]}" \
  --weights "${WEIGHTS[@]}" \
  --out_dir $FUSED

echo "========== [STEP 3] Evaluate all tasks =========="
python evaluate_all_tasks.py \
  --model $FUSED \
  --base_model $BASE \
  --max_examples 200 > recall_results_500.txt

echo "========== [STEP 4] Analyze results =========="
python analyze_results.py

echo "✅ Finished small-sample RECALL run (subset=${SUBSET})"
echo "Results saved to recall_results_500.txt and summary_table.csv"

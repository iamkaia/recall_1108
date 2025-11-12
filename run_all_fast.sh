#!/bin/bash
set -e

# =============================
# ⚡ RECALL (5hr fast reproduction)
# =============================

BASE_MODEL="Qwen/Qwen2-7B-Instruct"
OUT_DIR="checkpoints_fast"
FUSED_DIR="fused_recall_fast"

# Fast config
SUBSET=20000
EPOCHS=3
TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")

echo "========== [STEP 0] Environment setup =========="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT_DIR

echo "========== [STEP 1] Train each task (subset=$SUBSET, epochs=$EPOCHS) =========="
for TASK in "${TASKS[@]}"; do
    CKPT_PATH="$OUT_DIR/$TASK"
    if [ -d "$CKPT_PATH" ] && [ -f "$CKPT_PATH/adapter_config.json" ]; then
        echo "✅ Skip $TASK, already trained at $CKPT_PATH"
    else
        echo "🚀 Training $TASK ..."
        python train_single_task.py \
            --base_model $BASE_MODEL \
            --task $TASK \
            --subset $SUBSET \
            --epochs $EPOCHS \
            --output_dir $OUT_DIR
    fi
done

echo "========== [STEP 2] Convert safetensors → bin if needed =========="
python - <<'PYCODE'
import os, torch
from safetensors.torch import load_file

tasks = ["sst2","squad2","iwslt2017","race","medmcqa"]
base_dir = "checkpoints_fast"
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

echo "========== [STEP 3] Merge adapters (RECALL fusion) =========="
python merge_recall.py \
  --base_model $BASE_MODEL \
  --adapters $OUT_DIR/sst2 $OUT_DIR/squad2 $OUT_DIR/iwslt2017 $OUT_DIR/race $OUT_DIR/medmcqa \
  --weights 0.9 0.8 0.8 0.7 0.7 \
  --out_dir $FUSED_DIR

echo "========== [STEP 4] Evaluate all tasks (test split, subset=$SUBSET) =========="
python evaluate_all_tasks.py \
  --model $FUSED_DIR \
  --base_model $BASE_MODEL \
  --split test \
  --max_examples $SUBSET > recall_results_fast.txt

echo "========== [STEP 5] Analyze results =========="
python analyze_results.py

echo "✅ Finished RECALL (Fast Mode, ~5hr total)"
echo "📂 Outputs:"
echo "   - $OUT_DIR/"
echo "   - $FUSED_DIR/"
echo "   - recall_results_fast.txt"
echo "   - summary_table.csv"

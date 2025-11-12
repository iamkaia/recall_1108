#!/bin/bash
set -e

# =============================
# ✅ FINAL RECALL (Reproduction of paper)
# =============================

BASE_MODEL="Qwen/Qwen2-7B"
TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")

OUT_DIR="checkpoints_200"
FUSED_DIR="fused_recall_200"
DATASETS_ROOT="./datasets"

EPOCHS=3        # 論文：多數 task 訓練 3~5 epochs
SUBSET=200        # 0 = 不抽樣，全量訓練

LOG_DIR="logs"
mkdir -p $OUT_DIR $LOG_DIR
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M).log"

echo "========== [STEP 0] Env setup ==========" | tee -a $LOG_FILE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------- 1️⃣ Train each task --------------------
echo "========== [STEP 1] Train each task ==========" | tee -a $LOG_FILE

for TASK in "${TASKS[@]}"; do
    CKPT_PATH="$OUT_DIR/$TASK"

    if [ -d "$CKPT_PATH" ] && [ -f "$CKPT_PATH/adapter_config.json" ]; then
        echo "✅ Skip (Already exists): $TASK" | tee -a $LOG_FILE
    else
        echo "🚀 Training task: $TASK" | tee -a $LOG_FILE
        python train_single_task.py \
            --base_model $BASE_MODEL \
            --task $TASK \
            --subset $SUBSET \
            --epochs $EPOCHS \
            --output_dir $OUT_DIR \
            2>&1 | tee -a $LOG_FILE
    fi
done

# -------------------- 2️⃣ Convert safetensors → bin --------------------
echo "========== [STEP 2] Ensure adapter_model.bin exists ==========" | tee -a $LOG_FILE
python << 'EOF'
import os, torch
from safetensors.torch import load_file

root = "checkpoints_full"
for task in ["sst2", "squad2", "iwslt2017", "race", "medmcqa"]:
    p = f"{root}/{task}/adapter_model.safetensors"
    if os.path.exists(p):
        bin_p = f"{root}/{task}/adapter_model.bin"
        if not os.path.exists(bin_p):
            print(f"⚙️  Converting {p} → adapter_model.bin")
            torch.save(load_file(p), bin_p)
EOF

# -------------------- 3️⃣ Merge using RECALL (Layer-wise SVD) --------------------
echo "========== [STEP 3] Merge adapters (SVD / RECALL) ==========" | tee -a $LOG_FILE

python merge_recall.py \
  --base_model $BASE_MODEL \
  --adapters_root $OUT_DIR \
  --out_dir $FUSED_DIR \
  2>&1 | tee -a $LOG_FILE



# -------------------- 4️⃣ Evaluate all tasks --------------------
echo "========== [STEP 4] Evaluate fused model ==========" | tee -a $LOG_FILE

python evaluate_all_tasks_paper.py \
  --model $FUSED_DIR \
  --tokenizer $BASE_MODEL \
  --datasets_root ./datasets \
  2>&1 | tee -a $LOG_FILE



# -------------------- 5️⃣ Summary --------------------
echo "========== [STEP 5] Generate avg score ==========" | tee -a $LOG_FILE
python analyze_results.py | tee -a $LOG_FILE

echo "✅ Finished RECALL reproduction" | tee -a $LOG_FILE
echo "📂 Check outputs:" | tee -a $LOG_FILE
echo "   - $OUT_DIR/" | tee -a $LOG_FILE
echo "   - $FUSED_DIR/" | tee -a $LOG_FILE
echo "   - $LOG_FILE" | tee -a $LOG_FILE

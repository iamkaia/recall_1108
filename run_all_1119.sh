#!/usr/bin/env bash
set -euo pipefail

# =============================
# RECALL full pipeline (4090 single-GPU, QLoRA)
# - base: Qwen/Qwen2-7B-Instruct (等價於 LLaMA-2-7B-Chat 的 chat/instruct基礎)
# - 可用 N_SAMPLES 控制每個 task 取多少筆 (0 表示全量)
# - epochs/batch 依 Table 5 調整（再經單卡化：per_device_batch=1 + GA 累積在 train_single_task.py 內處理）
# =============================

# logs1116也是你的
# ---- 可調參數 ----
#BASE_MODEL="Qwen/Qwen2-7B-Instruct"
#OUT_DIR="checkpoints_recall"
#FUSED_DIR="fused_recall_qwen2"
#BASE_MODEL="./Llama-2-7b-chat-hf"
BASE_MODEL="unsloth/llama-2-7b-chat"
FUSED_DIR="fused_recall_1119_test"
OUT_DIR="checkpoints_recall_1119_test"
DATA_ROOT="./datasets"                  # 你的 jsonl 所在根目錄
RESULTS_DIR="results_eval_1119_test"
LOG_DIR="logs"

# 控制每個 task 訓練只抽部分資料（0=全量）
N_SAMPLES="${N_SAMPLES:-200}"           # 例：export N_SAMPLES=200

# 生成/評估長度 (保守避免 OOM)
MAX_LEN_TRAIN=1024
MAX_LEN_EVAL=512
MAX_NEW_TOKENS=64

# 任務清單（論文五個）
TASKS=("sst2" "squad2" "iwslt2017" "race" "medmcqa")

mkdir -p "$OUT_DIR" "$FUSED_DIR" "$RESULTS_DIR" "$LOG_DIR"

TIME_TAG=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/run_${TIME_TAG}.log"

echo "========== [STEP 0] Env ==========" | tee -a "$LOG_FILE"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
# （可選）使用 Flash-Attn 2 時常見：export CUDA_VISIBLE_DEVICES=0

# -------------------------------
# 每個 task 的論文設定（Table 5），單卡化方案：
#   per_device_batch 固定 1；GA 在 train_single_task.py 內按 task 自動設定
#   這邊只傳 epochs + subset（N_SAMPLES）
# -------------------------------
epochs_for() {
  case "$1" in
    sst2) echo 3;;
    squad2) echo 4;;
    medmcqa) echo 3;;
    race) echo 5;;
    iwslt2017) echo 5;;
    *) echo 3;;
  esac
}

echo "========== [STEP 1] Train each task (QLoRA, subset=${N_SAMPLES}) ==========" | tee -a "$LOG_FILE"

for TASK in "${TASKS[@]}"; do
  CKPT_DIR="$OUT_DIR/$TASK"
  if [[ -f "$CKPT_DIR/adapter_config.json" ]]; then
    echo "✅ Skip $TASK (exists: $CKPT_DIR)" | tee -a "$LOG_FILE"
    continue
  fi

  EPOCHS="$(epochs_for "$TASK")"
  echo "🚀 Training $TASK (epochs=$EPOCHS, subset=$N_SAMPLES) ..." | tee -a "$LOG_FILE"

  # 說明：
  # - 需要你使用「我給的 QLoRA 版 train_single_task.py」（支援 --load_in_4bit / --max_len / --subset / --epochs）
  # - 內部會依 task 自動設置 LoRA r/alpha/dropout & GA，以符合論文 Table 5 的等效 batch
  # python train_single_task_unsloth.py \
  #   --task "$TASK" \
  #   --base_model "$BASE_MODEL" \
  #   --output_dir "$OUT_DIR" \
  #   --subset "$N_SAMPLES" \
  #   --epochs "$EPOCHS" \
  #   --load_in_4bit \
  #   --max_len "$MAX_LEN_TRAIN" 2>&1 | tee -a "$LOG_FILE"
# done

  python train_single_task_unsloth.py \
    --task "$TASK" \
    --dataset_dir "$DATA_ROOT" \
    --output_dir "$OUT_DIR" \
    --base_model "$BASE_MODEL" \
    --subset "$N_SAMPLES" \
    --load_in_4bit \
    --max_len "$MAX_LEN_TRAIN" 2>&1 | tee -a "$LOG_FILE"
done

echo "✅ 所有LoRA training完成" | tee -a "$LOG_FILE"

# safetensors -> bin（若需要）
echo "========== [STEP 1.5] Ensure LoRA bin ==========" | tee -a "$LOG_FILE"
python - <<'PY'
import os, torch, glob
from safetensors.torch import load_file
root="checkpoints_recall"
for ad in glob.glob(os.path.join(root, "*")):
    if not os.path.isdir(ad): continue
    st=os.path.join(ad,"adapter_model.safetensors")
    bn=os.path.join(ad,"adapter_model.bin")
    if os.path.exists(st) and not os.path.exists(bn):
        print(f"⚙️  convert {st} -> {bn}")
        torch.save(load_file(st), bn)
print("done.")
PY

# ------------------------------- 
# RECALL merge（逐層 similarity + softmax） 
# * 需要「我的 merge_recall.py（layer-wise similarity + softmax）」版本 
# * 為避免 hidden 長度不齊導致 stack error，這裡固定 merge 時的收集長度 
# ------------------------------- 
echo "========== [STEP 2] RECALL merge (layer-wise similarity + softmax) ==========" | tee -a "$LOG_FILE" 
# 固定收集 hidden 的 prompt 長度，避免堆疊維度不一致 
export RECALL_MERGE_PADLEN="${RECALL_MERGE_PADLEN:-128}" 
# 收集每個 task 的小樣本作為表示對齊（每 task 20 條即可） 
export RECALL_MERGE_SAMPLES_PER_TASK="${RECALL_MERGE_SAMPLES_PER_TASK:-20}" 

python merge_recall.py \
  --base_model "$BASE_MODEL" \
  --adapters_root "$OUT_DIR" \
  --data_root "$DATA_ROOT" \
  --output_dir "$FUSED_DIR" 2>&1 | tee -a "$LOG_FILE"


#python merge_he_oa_recall.py \
#  --base_model "$BASE_MODEL" \
#  --adapters_root "$OUT_DIR" \
#  --data_root "$DATA_ROOT" \
#  --output_dir "$FUSED_DIR" \
#  --samples_per_task 20 \
#  --pad_len 128 \
# -------------------------------
# Evaluate（關閉 chat 模板，逐 task EM/Acc/BLEU）
# -------------------------------
echo "========== [STEP 3] Evaluate fused model ==========" | tee -a "$LOG_FILE"

python evaluate_all_tasks.py \
  --model "$FUSED_DIR" \
  --base_model "$BASE_MODEL" \
  --data_root "$DATA_ROOT" \
  --results_dir "$RESULTS_DIR" \
  --max_examples 100 \
  --max_src_len "$MAX_LEN_EVAL" \
  --max_new_tokens "$MAX_NEW_TOKENS" 2>&1 | tee -a "$LOG_FILE"

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

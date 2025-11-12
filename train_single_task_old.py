import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import argparse

# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="checkpoints_500")
parser.add_argument("--subset", type=int, default=200, help="Use 0 for full dataset")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default 1)")
args = parser.parse_args()

# ----------------------------
# Load dataset
# ----------------------------
task = args.task
if os.path.exists(f"datasets/{task}/train.jsonl"):
    from datasets import load_dataset
    data = load_dataset("json", data_files={
        "train": f"datasets/{task}/train.jsonl",
        "test": f"datasets/{task}/test.jsonl"
    })
else:
    print(f"❌ Local dataset not found for {task}. Please run prepare_datasets.py first.")
    exit(1)

train_data = data["train"]
if args.subset > 0:
    train_data = train_data.select(range(min(args.subset, len(train_data))))

# <== 在這裡放上新的 preprocess 區塊
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(example):
    if "target" in example:
        text = example["input"] + " " + example["target"]
    else:
        text = example["input"]
    model_inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

train_tokenized = train_data.map(preprocess, batched=False)

# ----------------------------
# Model with LoRA adapter
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ----------------------------
# Training setup
# ----------------------------
out_dir = os.path.join(args.out_dir, task)
os.makedirs(out_dir, exist_ok=True)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.03,
    num_train_epochs=args.epochs,
    learning_rate=2e-4,  # same as paper
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    output_dir=out_dir,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    tokenizer=tokenizer,
)

print(f"🚀 Start training task: {task}, epochs={args.epochs}, samples={len(train_tokenized)}")
trainer.train()

model.save_pretrained(out_dir)
print(f"✅ Saved LoRA adapter for {task} at {out_dir}")

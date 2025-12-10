#!/usr/bin/env python3
import argparse, os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ============================
# Table hyperparameter rules
# ============================
TABLE_CONFIG = {
    "sst2":      {"r": 8, "alpha": 32, "drop": 0.1, "batch": 64,  "epochs": 3},
    "squad2":    {"r": 8, "alpha": 32, "drop": 0.1, "batch": 32,  "epochs": 4},
    "medmcqa":   {"r": 8, "alpha": 32, "drop": 0.1, "batch": 64,  "epochs": 3},
    "race":      {"r": 8, "alpha": 32, "drop": 0.1, "batch": 128, "epochs": 5},
    "iwslt2017": {"r": 8, "alpha": 32, "drop": 0.1, "batch": 64,  "epochs": 5},
}

def show_debug_samples(dataset, n=3):
    print("\n🔍 DEBUG | Sample formatted data preview:\n" + "-"*60)
    for i in range(min(n, len(dataset))):
        print(f"\n📌 SAMPLE {i+1}:")
        print(dataset[i])
        print("-"*60)
    print("✔ Debug preview done.\n")

def apply_format(task, example):
    if task == "sst2":
        return {
            "text": (
                f"Instruction: Statement: {example['input']}\n"
                f"OPTIONS:\n- negative\n- positive\n"
                f"Answer:\nOutput: {example['target']}"
            )
        }

    elif task == "squad2":
        return {
            "text": (
                f"Instruction: Based on the passage, answer the question. "
                f"If impossible, reply 'impossible to answer.'\n"
                f"Passage: {example['input']}\n\n"
                f"Answer:\nOutput: {example['target']}"
            )
        }

    elif task == "medmcqa":
        return {
            "text": (
                f"Instruction: Choose the correct answer.\n"
                f"{example['input']}\n"
                f"Answer:\nOutput: {example['target']}"
            )
        }

    elif task == "race":
        return {
            "text": (
                f"Instruction: Read the article and choose the correct option.\n"
                f"{example['input']}\n"
                f"Answer:\nOutput: {example['target']}"
            )
        }

    elif task == "iwslt2017":
        return {
            "text": (
                f"Instruction: Translate English to French.\n"
                f"Sentence: {example['input']}\n"
                f"Answer:\nOutput: {example['target']}"
            )
        }

    else:
        return {"text": f"{example['input']}\nAnswer: {example['target']}"}



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--base_model", default="unsloth/Qwen2-7B-Instruct-bnb-4bit")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true")

    args = parser.parse_args()

    if args.task not in TABLE_CONFIG:
        raise ValueError(f"❌ Task {args.task} 不在 InsCL/RECALL 官方任務集！")

    cfg = TABLE_CONFIG[args.task]

    print(f"""
==============================
 🚀 Training Task: {args.task}
==============================
📁 Dataset = {args.dataset_dir}/{args.task}
🧠 Model   = {args.base_model}
📌 LoRA    = r {cfg['r']} / α {cfg['alpha']} / dropout {cfg['drop']}
📌 Batch   = {cfg['batch']}
📌 Epochs  = {cfg['epochs']}
🔧 Max Len = {args.max_len}
==============================\n""")


    # ---------------- Load JSONL Dataset ----------------
    path = f"{args.dataset_dir}/{args.task}"
    val_file = "validation.jsonl" if os.path.exists(f"{path}/validation.jsonl") else "test.jsonl"

    data = load_dataset("json", data_files={
        "train": f"{path}/train.jsonl",
        "validation": f"{path}/{val_file}",
    })

    # ---------------- Apply SFT text formatting ----------------
    data = data.map(lambda x: {"text": f"Instruction: {x['input']}\nOutput: {x['target']}"})
    #data = data.map(lambda x: apply_format(args.task, x))


    # ---------------- Optional Subset ----------------
    if args.subset > 0:
        data["train"] = data["train"].shuffle(seed=42).select(range(args.subset))
        print(f"📌 Subset enabled: {args.subset} samples")

    # ---------------- Debug: Show Input Preview ----------------
    show_debug_samples(data["train"])

    # ---------------- Load Model ----------------
    #tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_len,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["r"],
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["drop"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ---------------- Training Config ----------------
    save_dir = f"{args.output_dir}/{args.task}"
    training_config = SFTConfig(
        output_dir=save_dir,
        per_device_train_batch_size=max(1, cfg["batch"] // 32),  # 🚨 scaling for 4090
        gradient_accumulation_steps=32,  # simulate large batch
        num_train_epochs=cfg["epochs"],
        learning_rate=5e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        save_total_limit=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        args=training_config,
        dataset_text_field="text",
        packing=False,
    )

    trainer.train()
    #trainer.save_model(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"🎉 DONE — Final model saved at: {save_dir}\n")


if __name__ == "__main__":
    main()

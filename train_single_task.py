import os, argparse, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# -------------------- Args --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--base_model", type=str, default="./Llama-2-7b-chat-hf")
parser.add_argument("--output_dir", default="checkpoints_recall")
parser.add_argument("--subset", type=int, default=0)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_len", type=int, default=384)
parser.add_argument("--load_in_4bit", action="store_true")
args = parser.parse_args()

# -------------------- Dataset --------------------
data = load_dataset("json", data_files={
    "train": f"datasets/{args.task}/train.jsonl"
})
if args.subset > 0:
    data["train"] = data["train"].select(range(min(args.subset, len(data["train"]))))
print(f"🚀 Train {args.task} ({len(data['train'])} samples, {args.epochs} epochs)")

# -------------------- Tokenizer --------------------
tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

def preprocess(ex):
    p, a = ex["input"], ex["target"]
    full = f"{p}\nAnswer: {a}"
    pref = f"{p}\nAnswer:"
    t = tok(full, truncation=True, max_length=args.max_len, add_special_tokens=True)
    pref_ids = tok(pref, truncation=True, max_length=args.max_len, add_special_tokens=True)["input_ids"]
    labels = t["input_ids"][:]
    labels[:len(pref_ids)] = [-100]*len(pref_ids)
    t["labels"] = labels
    return t

train_tok = data["train"].map(preprocess, remove_columns=data["train"].column_names)
#test_tok  = data["test"].map(preprocess, remove_columns=data["test"].column_names)

# -------------------- Model (QLoRA) --------------------
bnb_cfg = None
m_kwargs = {"device_map": "auto"}
if args.load_in_4bit:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    m_kwargs["quantization_config"] = bnb_cfg
else:
    m_kwargs["torch_dtype"] = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(args.base_model, **m_kwargs)

try:
    model.config.attn_implementation = "flash_attention_2"
except Exception:
    pass

lora_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# -------------------- Trainer --------------------
collator = DataCollatorForSeq2Seq(tok, model=None, padding=True, label_pad_token_id=-100)
train_args = TrainingArguments(
    output_dir=f"{args.output_dir}/{args.task}",
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=8,
    num_train_epochs=args.epochs,
    learning_rate=5e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    bf16=True,
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
)

trainer = Trainer(
    model=model, args=train_args,
    train_dataset=train_tok,
    tokenizer=tok, data_collator=collator,
)

trainer.train()
model.save_pretrained(f"{args.output_dir}/{args.task}")
tok.save_pretrained(f"{args.output_dir}/{args.task}")
print(f"✅ Done: {args.task}")

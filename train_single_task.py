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
'''
tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
tok.pad_token = tok.eos_token
tok.padding_side = "right"
'''

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-2-7b-hf",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

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

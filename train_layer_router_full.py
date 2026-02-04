import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from sklearn.metrics import accuracy_score, f1_score


# =========================
# Label space (must match experts)
# =========================
ID2LABEL = {
    0: "iwslt2017",
    1: "medmcqa",
    2: "race",
    3: "squad2",
    4: "sst2",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# =========================
# Utils
# =========================
def can_load_locally(model_name_or_path: str) -> bool:
    try:
        AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
        return True
    except Exception:
        return False


def load_split(data_dir, split, text_key="text"):
    texts, labels = [], []

    for task in sorted(os.listdir(data_dir)):
        if task not in LABEL2ID:
            continue
        path = os.path.join(data_dir, task, f"{split}.jsonl")
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                texts.append(ex[text_key])
                labels.append(LABEL2ID[task])

    return texts, labels


# =========================
# Model: Frozen LLaMA + Layer Routers
# =========================
class LayerRouterModel(nn.Module):
    """
    Output logits shape: [B, L, E]
    Loss: CE averaged over layers
    """
    def __init__(self, base_model, num_experts):
        super().__init__()
        self.base = base_model
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.num_layers = len(self.base.model.layers)
        hidden = self.base.config.hidden_size

        self.routers = nn.ModuleList([
            nn.Linear(hidden, num_experts, bias=False)
            for _ in range(self.num_layers)
        ])
        for r in self.routers:
            nn.init.zeros_(r.weight)

        self.ce = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        with torch.no_grad():
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hs = out.hidden_states  # [L+1][B,T,H]

        pooled = [hs[l].mean(dim=1) for l in range(1, self.num_layers + 1)]

        logits_layers = []
        for l in range(self.num_layers):
            x = pooled[l].float()
            logits_layers.append(self.routers[l](x))

        logits = torch.stack(logits_layers, dim=1)  # [B,L,E]

        loss = None
        if labels is not None:
            loss = sum(self.ce(logits[:, l], labels) for l in range(self.num_layers))
            loss = loss / self.num_layers

        return {"loss": loss, "logits": logits}


# =========================
# Metrics (對齊 external router)
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: [N,L,E]
    last_logits = logits[:, -1, :]
    preds = np.argmax(last_logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="datasets_classifier")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--out_dir", default="layer_router_full_ckpt")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train_bs", type=int, default=1)
    ap.add_argument("--eval_bs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_only", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------- data ----------
    train_texts, train_labels = load_split(args.data_dir, "train", args.text_key)
    val_texts, val_labels = load_split(args.data_dir, "validation", args.text_key)

    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
        "validation": Dataset.from_dict({"text": val_texts, "label": val_labels}),
    })

    # ---------- tokenizer ----------
    local_files_only = args.local_only or can_load_locally(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # ---------- base model ----------
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        local_files_only=local_files_only,
    )

    model = LayerRouterModel(base, num_experts=len(ID2LABEL))

    # ---------- training ----------
    common_kwargs = dict(
        output_dir=args.out_dir,
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        fp16=True,
    )

    try:
        # 舊版 transformers
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            **common_kwargs,
        )
    except TypeError:
        # 新版 transformers（你現在這個）
        training_args = TrainingArguments(
            eval_strategy="epoch",
            **common_kwargs,
        )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print("✅ Layer-router training finished (full train.jsonl).")


if __name__ == "__main__":
    main()

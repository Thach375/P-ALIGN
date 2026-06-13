"""
weighted_trainer.py

WeightedSFTDataset, WeightedDataCollator, WeightedSFTTrainer for Plan B.
Used by nb02_phase_b_sft.ipynb.
"""

import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer

MAX_LEN = 2048


class WeightedSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_len: int = MAX_LEN):
        with open(path) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        tok = self.tokenizer

        messages = [
            {"role": "user",      "content": f"Question: {row['question']}"},
            {"role": "assistant", "content": row["full_sequence"]},
        ]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = tok(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        # Mask user turn: predict only assistant response tokens
        prompt_only = tok.apply_chat_template(
            [{"role": "user", "content": f"Question: {row['question']}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = len(tok(prompt_only, return_tensors="pt")["input_ids"][0])
        labels[:prompt_len]              = -100
        labels[attention_mask == 0]      = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "weight":         torch.tensor(row["weight"], dtype=torch.float32),
        }


@dataclass
class WeightedDataCollator:
    def __call__(self, features: list) -> dict:
        keys  = ["input_ids", "attention_mask", "labels"]
        batch = {k: torch.stack([f[k] for f in features]) for k in keys}
        batch["weights"] = torch.stack([f["weight"] for f in features])
        return batch


class WeightedSFTTrainer(Trainer):
    """
    Length-normalized NLL loss weighted by per-sample advantage weights.
    Loss = sum_i( w_i * NLL_i / len_i ) / sum_i(w_i)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights")          # [B]
        outputs = model(**inputs)
        logits  = outputs.logits                 # [B, T, V]
        labels  = inputs["labels"]               # [B, T]
        B, T, V = logits.shape

        shift_logits = logits[:, :-1, :].contiguous().view(-1, V)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        token_loss   = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        token_loss   = token_loss.view(B, T - 1)          # [B, T-1]

        valid      = (labels[:, 1:] != -100).float()       # [B, T-1]
        per_sample = (token_loss * valid).sum(-1) / (valid.sum(-1) + 1e-6)  # [B]

        w = weights.to(per_sample.device)
        loss = (per_sample * w).sum() / (w.sum() + 1e-6)

        return (loss, outputs) if return_outputs else loss

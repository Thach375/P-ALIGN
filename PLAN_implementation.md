# Implementation Plan: Reward-Weighted Prefix-Alignment Distillation

**Target:** Implement Plan B (plan-B-reward-weighted-prefix-alignment.md) as a runnable
experiment on Google Colab T4 (15 GB VRAM).  
**Student model:** Qwen2.5-1.5B-Instruct  
**Training stack:** TRL SFTTrainer + PEFT LoRA  
**Scope:** Phase A (data building) + Phase B (reward-weighted SFT). DPO pairs are
collected during Phase A but DPO training is deferred.  
**Delivery:** 3 separate notebooks + updated utility scripts.

---

## Data Flow

```
[HF: qizheyanger/P-ALIGN]          [HF: simplescaling/s1K-1.1]
  instruction, input, output    +     question, deepseek_thinking_trajectory, solution
  (Alpaca format; instruction         (full DeepSeek-R1 CoT + ground-truth answer)
   encodes *Question* + *Prefix*)
          │                                        │
          └──── parse + join on question text ─────┘
                        │
                        ▼  Drive: prefixes.jsonl
                  question, prefix, answer, full_cot
                  (full_cot = deepseek_thinking_trajectory,
                   used only by prefix-feedback loop Step 6)
                                              │
                              Notebook 1     ▼
                              ─────────────────────────────────────────
                              Step 3: sample K=4 continuations per row
                              Step 4: score each with math verifier
                              Step 5: compute group-relative advantage
                              Step 6: prefix-feedback loop (τ_low=0.2)
                              Build SFT dataset (L1 weights + L2 weights)
                              Build DPO pairs  (for later)
                              ─────────────────────────────────────────
                                     │ Drive: sft_data_L1.jsonl
                                     │        sft_data_L2.jsonl
                                     │        dpo_pairs.jsonl
                              Notebook 2     ▼
                              ─────────────────────────────────────────
                              Load data → custom WeightedSFTDataset
                              WeightedSFTTrainer (length-normalized NLL)
                              LoRA fine-tune (3 epochs, lr=5e-5)
                              Save adapter → Drive/checkpoints/sft_L1/
                                                              sft_L2/
                                                              sft_L2_pf/
                              ─────────────────────────────────────────
                                     │
                              Notebook 3     ▼
                              ─────────────────────────────────────────
                              Load adapter → base + LoRA
                              Inference on AIME24 / AIME25 / AMC12
                              Evaluate: pass@1, pass@3, acc@3
                              Print ablation table
                              ─────────────────────────────────────────
```

---

## T4 Constraints & Mitigations

| Concern | Mitigation |
|---|---|
| VRAM for generation (K=4) | Generate continuations sequentially, not batched K at once; Qwen2.5-1.5B in fp16 ≈ 3 GB, leaves ~11 GB for KV cache |
| VRAM for LoRA training | batch=4, grad_accum=8 → effective 32; fp16; gradient checkpointing ON |
| Session timeout | All intermediate files go to `/content/drive/MyDrive/P-ALIGN/`; every notebook is resumable via skip-if-exists logic |
| Binary search cost | Pre-computed prefixes come from `qizheyanger/P-ALIGN` (Alpaca-format, parsed via regex). Ground-truth answers and full CoTs come from `simplescaling/s1K-1.1`, joined on question text. No binary search re-run needed. |

---

## File Layout After Implementation

```
P-ALIGN/
├── notebooks/
│   ├── nb01_phase_a_data_building.ipynb
│   ├── nb02_phase_b_sft.ipynb
│   └── nb03_evaluation.ipynb
├── src/
│   ├── binary_search.py          ← unchanged
│   ├── prefix-alignment.py       ← unchanged (single-sample generation, kept for reference)
│   ├── evaluation.py             ← unchanged
│   ├── reward_weighted_data.py   ← NEW: Steps 3-6 logic (pure Python, importable)
│   └── weighted_trainer.py       ← NEW: WeightedSFTDataset + WeightedSFTTrainer
├── scripts/
│   └── (existing scripts unchanged)
└── PLAN-B-implementation.md      ← this file
```

---

## Notebook 1 — `nb01_phase_a_data_building.ipynb`

Goal: produce `sft_data_L1.jsonl`, `sft_data_L2.jsonl`, `dpo_pairs.jsonl` on Drive.

### Cell 1 — Mount Drive & Install Dependencies

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers accelerate tqdm datasets math-verify sympy peft

BASE_DIR = "/content/drive/MyDrive/P-ALIGN"
import os; os.makedirs(f"{BASE_DIR}/data", exist_ok=True)
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
```

### Cell 2 — Download Model (skip if already on Drive)

```python
from huggingface_hub import snapshot_download
MODEL_DIR = f"{BASE_DIR}/models/Qwen2.5-1.5B-Instruct"
if not os.path.exists(f"{MODEL_DIR}/config.json"):
    snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir=MODEL_DIR)
```

### Cell 3 — Load & Join Datasets → `prefixes.jsonl`

```python
import re, json
from datasets import load_dataset

PREFIX_FILE = f"{BASE_DIR}/data/prefixes.jsonl"

if os.path.exists(PREFIX_FILE):
    with open(PREFIX_FILE) as f:
        rows = [json.loads(l) for l in f]
    print(f"[Resume] Loaded {len(rows)} rows from {PREFIX_FILE}")
else:
    # ── Step A: parse P-ALIGN (Alpaca format) ─────────────────────────────────
    # instruction = full prompt encoding "*Question*: ...\n*Prefix*: ..."
    # input       = empty (standard Alpaca)
    # output      = target continuation (not used here; we re-generate in Step 3)
    palign = load_dataset("qizheyanger/P-ALIGN", split="train")
    print(f"P-ALIGN columns: {palign.column_names}")  # confirm: instruction, input, output

    _Q_RE = re.compile(
        r"\*Question\*\s*[:\-]?\s*(.*?)\s*\*Prefix\*\s*[:\-]?\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )

    def parse_instruction(instruction: str):
        m = _Q_RE.search(instruction)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # Fallback: whole instruction is the question, no prefix
        return instruction.strip(), ""

    palign_parsed = []
    for item in palign:
        q, p = parse_instruction(item["instruction"])
        palign_parsed.append({"question": q, "prefix": p})

    # ── Step B: load s1K-1.1 for full CoT + ground-truth answer ───────────────
    # Columns: question, deepseek_thinking_trajectory, solution, ...
    s1k = load_dataset("simplescaling/s1K-1.1", split="train")
    print(f"s1K-1.1 columns: {s1k.column_names}")  # confirm exact names

    s1k_lookup = {}   # question_text → {full_cot, answer}
    for item in s1k:
        key = item["question"].strip()
        s1k_lookup[key] = {
            "full_cot": item["deepseek_thinking_trajectory"],
            "answer":   str(item["solution"]).strip(),
        }
    print(f"s1K-1.1 lookup built: {len(s1k_lookup)} entries")

    # ── Step C: join on question text ─────────────────────────────────────────
    rows, n_miss = [], 0
    for i, parsed in enumerate(palign_parsed):
        q = parsed["question"]
        lookup = s1k_lookup.get(q)
        if lookup is None:
            # Fuzzy fallback: strip punctuation and lowercase
            q_norm = re.sub(r"\s+", " ", q.lower().strip())
            lookup = next(
                (v for k, v in s1k_lookup.items()
                 if re.sub(r"\s+", " ", k.lower().strip()) == q_norm),
                None,
            )
        if lookup is None:
            n_miss += 1
            continue
        rows.append({
            "id":       i,
            "question": q,
            "prefix":   parsed["prefix"],
            "answer":   lookup["answer"],
            "full_cot": lookup["full_cot"],   # deepseek_thinking_trajectory
        })

    print(f"Joined {len(rows)} samples ({n_miss} unmatched, dropped)")

    with open(PREFIX_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved → {PREFIX_FILE}")
```

> **Column names confirmed (2025-06-14):**  
> `qizheyanger/P-ALIGN` → `instruction`, `input`, `output`  
> `simplescaling/s1K-1.1` → `question`, `deepseek_thinking_trajectory`, `solution`
>
> The `print(…column_names)` lines above will alert you if these ever change.  
> If `n_miss` is large (> 5 %), the prefix whitespace or encoding differs — add a
> `unicodedata.normalize("NFC", q)` step to both sides of the join key.

### Cell 4 — Load Student Model for Generation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
```

### Cell 5 — Hyperparameters

```python
K          = 4          # continuations per question
TEMP       = 0.8        # sampling temperature
TOP_P      = 0.95
MAX_NEW    = 2048       # max continuation tokens
TAU_LOW    = 0.2        # prefix-feedback threshold
T_MAX      = 3          # max prefix-extension attempts
LAMBDA_LEN = 0.0        # length penalty (set > 0 to enable shaping)
EPSILON    = 1e-6       # advantage denominator stabilizer
CLIP_C     = 3.0        # advantage clip for L2
```

### Cell 6 — Prompt Builder

```python
def build_prompt(question: str, prefix: str) -> str:
    return (
        "Please continue from the draft and solve the problem step by step, "
        "and put your final answer within \\boxed{}. "
        "I will provide you with some prior knowledge as a draft to assist you.\n"
        f"*Question*: {question}\n"
        f"*Prefix*: {prefix}"
    )
```

### Cell 7 — K-Continuation Sampler (Step 3)

```python
def sample_k_continuations(question: str, prefix: str, k: int = K) -> list[str]:
    """Generate K independent continuations from (question, prefix)."""
    prompt = build_prompt(question, prefix)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    continuations = []
    for _ in range(k):
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=TEMP,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
        cont = tokenizer.decode(
            out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        continuations.append(cont)
    
    # Diversity check: warn if all K answers are identical
    from collections import Counter
    answers = [extract_boxed(c) for c in continuations]
    if len(Counter(answers)) == 1:
        print(f"  ⚠️  All K continuations identical — consider raising temperature.")
    return continuations
```

### Cell 8 — Verifier (Step 4)

```python
import re, signal
from math_verify import parse, verify

def extract_boxed(text: str) -> str:
    """Extract last \\boxed{...} content."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1] if matches else ""

def verify_answer(pred: str, gold: str, timeout_sec: int = 5) -> int:
    """Return 1 if pred matches gold, else 0. Wraps sympy verify with timeout."""
    def _handler(s, f): raise TimeoutError
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        pred_ans = extract_boxed(pred)
        if not pred_ans: return 0
        p = parse(pred_ans); g = parse("$" + str(gold) + "$")
        return int(bool(verify(g, p)))
    except Exception:
        return 0
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)

def compute_rewards(continuations: list[str], answer: str) -> list[float]:
    """Step 4: binary reward ± optional length penalty."""
    rewards = []
    for y in continuations:
        r = float(verify_answer(y, answer))
        if LAMBDA_LEN > 0:
            r -= LAMBDA_LEN * len(y.split())
        rewards.append(r)
    return rewards
```

### Cell 9 — Group-Relative Advantage (Step 5)

```python
import math

def compute_advantage(rewards: list[float]) -> list[float]:
    """Step 5: group-normalize rewards into advantages."""
    mu = sum(rewards) / len(rewards)
    variance = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(variance) + EPSILON
    return [(r - mu) / sigma for r in rewards]

def classify_group(rewards: list[float]) -> str:
    """Return 'all_correct', 'all_wrong', or 'mixed'."""
    binary = [1 if r > 0 else 0 for r in rewards]
    if all(b == 1 for b in binary): return "all_correct"
    if all(b == 0 for b in binary): return "all_wrong"
    return "mixed"
```

### Cell 10 — Prefix-Feedback Loop (Step 6)

```python
def sentence_split(text: str) -> list[str]:
    """
    Simple sentence splitter mirroring binary_search.py's split_sentences.
    Splits on '. ' and re-attaches the period.
    """
    parts = text.split(". ")
    sentences = []
    for s in parts:
        s = s.strip()
        if s:
            sentences.append(s if s.endswith(".") else s + ".")
    return sentences

def build_prefix_ladder(row: dict) -> list[str]:
    """
    Return cumulative sentence-level prefixes from the full DeepSeek-R1 CoT.
    row["full_cot"] = deepseek_thinking_trajectory from s1K-1.1.
    Returns ["sent1.", "sent1. sent2.", ..., full_cot].
    The current prefix is always somewhere in this list; extension = next longer entry.
    """
    full_cot = row.get("full_cot", "")
    sentences = sentence_split(full_cot)
    return [" ".join(sentences[:i+1]) for i in range(len(sentences))]

def get_next_prefix(current_prefix: str, ladder: list[str]) -> str | None:
    """Return the first ladder entry strictly longer than current_prefix, or None."""
    cur_len = len(current_prefix)
    for candidate in ladder:
        if len(candidate) > cur_len:
            return candidate
    return None
```

### Cell 11 — Main Data-Building Loop

```python
import json
from tqdm import tqdm

SFT_L1_FILE  = f"{BASE_DIR}/data/sft_data_L1.jsonl"
SFT_L2_FILE  = f"{BASE_DIR}/data/sft_data_L2.jsonl"
DPO_FILE     = f"{BASE_DIR}/data/dpo_pairs.jsonl"
STATS_FILE   = f"{BASE_DIR}/data/build_stats.json"

# Resume support: load already-processed IDs
processed_ids = set()
for fpath in [SFT_L1_FILE, SFT_L2_FILE]:
    if os.path.exists(fpath):
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                processed_ids.add(d.get("source_id", ""))

with open(PREFIX_FILE) as f:
    rows = [json.loads(l) for l in f]

stats = {"total": 0, "all_correct": 0, "all_wrong": 0, "mixed": 0, "feedback_triggered": 0}

f_l1  = open(SFT_L1_FILE,  "a", encoding="utf-8")
f_l2  = open(SFT_L2_FILE,  "a", encoding="utf-8")
f_dpo = open(DPO_FILE,     "a", encoding="utf-8")

try:
  for row in tqdm(rows, desc="Building dataset"):
    src_id   = str(row.get("id", row["question"][:40]))
    if src_id in processed_ids:
        continue

    question = row["question"]
    prefix   = row["prefix"]
    answer   = str(row["answer"])
    stats["total"] += 1

    # Build the prefix extension ladder once per question (uses full_cot from s1K-1.1)
    ladder = build_prefix_ladder(row)

    # Step 6: prefix-feedback loop
    for attempt in range(T_MAX):
        continuations = sample_k_continuations(question, prefix, K)
        rewards       = compute_rewards(continuations, answer)
        pass_rate     = sum(1 for r in rewards if r > 0) / K

        if pass_rate < TAU_LOW:
            next_p = get_next_prefix(prefix, ladder)
            if next_p:
                prefix = next_p
                stats["feedback_triggered"] += 1
                continue
        break  # pass_rate >= TAU_LOW, or no further extension available

    group_type = classify_group(rewards)
    stats[group_type] = stats.get(group_type, 0) + 1

    if group_type == "all_wrong":
        continue  # no usable signal

    advantages = compute_advantage(rewards)

    # Build SFT samples
    mu_r = sum(rewards) / len(rewards)
    sum_w_l1 = sum(1.0 for r in rewards if r >= mu_r) + EPSILON
    sum_w_l2 = sum(max(advantages[k], 0) for k in range(K)) + EPSILON

    for k, (y, r, a_k) in enumerate(zip(continuations, rewards, advantages)):
        full_seq = prefix + "\n" + y

        # L1 weight: above-average binary
        w_l1 = (1.0 / sum_w_l1) if r >= mu_r else 0.0
        # L2 weight: clipped advantage, normalized
        w_l2 = (min(max(a_k, 0), CLIP_C) / sum_w_l2)

        base = {"source_id": src_id, "question": question,
                "prefix": prefix, "continuation": y,
                "full_sequence": full_seq, "answer": answer,
                "reward": r, "pass_rate": pass_rate}

        if w_l1 > 0:
            f_l1.write(json.dumps({**base, "weight": w_l1}, ensure_ascii=False) + "\n")
        if w_l2 > 0:
            f_l2.write(json.dumps({**base, "weight": w_l2}, ensure_ascii=False) + "\n")

    # DPO pairs (mixed groups only)
    if group_type == "mixed":
        best_k  = max(range(K), key=lambda k: rewards[k])
        worst_k = min(range(K), key=lambda k: rewards[k])
        if rewards[best_k] > rewards[worst_k]:
            f_dpo.write(json.dumps({
                "source_id": src_id, "question": question, "prefix": prefix,
                "chosen": continuations[best_k], "rejected": continuations[worst_k],
                "reward_chosen": rewards[best_k], "reward_rejected": rewards[worst_k],
            }, ensure_ascii=False) + "\n")

finally:
    f_l1.close(); f_l2.close(); f_dpo.close()

with open(STATS_FILE, "w") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))
```

### Cell 12 — Sanity Check & Stats

```python
# Count samples per file, print pass-rate distribution, average prefix ratio
import json

for label, fpath in [("L1", SFT_L1_FILE), ("L2", SFT_L2_FILE), ("DPO", DPO_FILE)]:
    with open(fpath) as f:
        lines = f.readlines()
    print(f"{label}: {len(lines)} samples in {fpath}")

with open(STATS_FILE) as f:
    print(json.load(f))
```

---

## Notebook 2 — `nb02_phase_b_sft.ipynb`

Goal: fine-tune Qwen2.5-1.5B-Instruct with weighted SFT. Run once for L1, once for L2
(and optionally once for L2 + prefix-feedback data).

### Cell 1 — Mount Drive & Install

```python
from google.colab import drive; drive.mount('/content/drive')
!pip install -q transformers accelerate peft trl bitsandbytes

BASE_DIR   = "/content/drive/MyDrive/P-ALIGN"
MODEL_DIR  = f"{BASE_DIR}/models/Qwen2.5-1.5B-Instruct"
```

### Cell 2 — Config Switch

```python
# ── CHANGE THIS FOR EACH ABLATION RUN ──────────────────────────────────────
ABLATION     = "L2"           # "L1" | "L2" | "L2_pf"
DATA_FILE    = f"{BASE_DIR}/data/sft_data_{ABLATION}.jsonl"
CKPT_DIR     = f"{BASE_DIR}/checkpoints/sft_{ABLATION}"
# ───────────────────────────────────────────────────────────────────────────
import os; os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Training {ABLATION} → {CKPT_DIR}")
```

### Cell 3 — Load & Tokenize Dataset

```python
import json, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
MAX_LEN = 2048

class WeightedSFTDataset(Dataset):
    def __init__(self, path: str):
        with open(path) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        # Instruction format: user message = prompt, assistant = full_sequence
        messages = [
            {"role": "user",      "content": f"Question: {row['question']}"},
            {"role": "assistant", "content": row["full_sequence"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(text, max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels: -100 on padding and user part, predict only assistant part
        labels = input_ids.clone()
        # Find where assistant response starts (after last <|im_start|>assistant token)
        # Simple heuristic: mask user turn via the prompt-only encoding length
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Question: {row['question']}"}],
            tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer(prompt_only, return_tensors="pt")["input_ids"][0])
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "weight":         torch.tensor(row["weight"], dtype=torch.float32),
        }

dataset = WeightedSFTDataset(DATA_FILE)
print(f"Dataset size: {len(dataset)}")
```

### Cell 4 — Data Collator

```python
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class WeightedDataCollator:
    def __call__(self, features: list[dict]) -> dict[str, Any]:
        keys = ["input_ids", "attention_mask", "labels"]
        batch = {k: torch.stack([f[k] for f in features]) for k in keys}
        batch["weights"] = torch.stack([f["weight"] for f in features])
        return batch
```

### Cell 5 — Custom Trainer (Weighted NLL Loss)

```python
from transformers import Trainer
import torch.nn.functional as F

class WeightedSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights")           # [B]
        outputs = model(**inputs)
        logits  = outputs.logits                  # [B, T, V]
        labels  = inputs["labels"]                # [B, T]
        B, T, V = logits.shape

        shift_logits = logits[:, :-1, :].contiguous().view(-1, V)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        token_loss   = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        token_loss   = token_loss.view(B, T - 1)          # [B, T-1]

        valid = (labels[:, 1:] != -100).float()            # [B, T-1]
        per_sample = (token_loss * valid).sum(-1) / (valid.sum(-1) + 1e-6)  # [B]

        loss = (per_sample * weights.to(per_sample.device)).sum() \
               / (weights.sum() + 1e-6)
        return (loss, outputs) if return_outputs else loss
```

### Cell 6 — Load Model + LoRA

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, device_map="auto"
)
base_model.config.use_cache = False  # required for gradient checkpointing

lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()
model.enable_input_require_grads()
```

### Cell 7 — Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=CKPT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,     # effective batch = 32
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    dataloader_num_workers=0,
    report_to="none",
    remove_unused_columns=False,       # keep "weights" column
)
```

### Cell 8 — Train

```python
trainer = WeightedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=WeightedDataCollator(),
)
trainer.train()
trainer.save_model(CKPT_DIR)
tokenizer.save_pretrained(CKPT_DIR)
print(f"✅ Saved to {CKPT_DIR}")
```

> **Repeat:** Change `ABLATION = "L1"` in Cell 2 and re-run from Cell 2 to train the L1 variant.  
> **Baseline (P-ALIGN original):** Train with `ABLATION = "L1"` but set all weights=1.0
> (i.e., standard SFT on correct-only samples, exactly what binary filtering does).

---

## Notebook 3 — `nb03_evaluation.ipynb`

Goal: evaluate each checkpoint and produce the ablation comparison table.

### Cell 1 — Setup

```python
from google.colab import drive; drive.mount('/content/drive')
!pip install -q transformers peft math-verify sympy tqdm

BASE_DIR  = "/content/drive/MyDrive/P-ALIGN"
MODEL_DIR = f"{BASE_DIR}/models/Qwen2.5-1.5B-Instruct"
EVAL_DIR  = f"{BASE_DIR}/eval"
import os; os.makedirs(EVAL_DIR, exist_ok=True)
```

### Cell 2 — Config

```python
# Pick which checkpoint to evaluate
CHECKPOINT = f"{BASE_DIR}/checkpoints/sft_L2"
N_SAMPLES  = 3     # continuations per question for pass@N
TEMP       = 0.6
TOP_P      = 0.9
MAX_NEW    = 4096
```

### Cell 3 — Load Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, CHECKPOINT)
model.eval()
```

### Cell 4 — Load Eval Benchmarks

```python
# Use local result files as reference questions, or load from HuggingFace
import json

def load_benchmark(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

benchmarks = {
    "AIME24": load_benchmark(f"{BASE_DIR}/../data/result/aime24-output.jsonl"),
    "AIME25": load_benchmark(f"{BASE_DIR}/../data/result/aime25-output.jsonl"),
    "AMC12":  load_benchmark(f"{BASE_DIR}/../data/result/amc12-output.jsonl"),
}
# These files contain {"question": ..., "answer": ...} — just need q/a for inference
```

### Cell 5 — Inference

```python
from tqdm import tqdm

def generate_answers(question: str, n: int = N_SAMPLES) -> list[str]:
    prompt = (
        "Solve the following math problem step by step. "
        "Put your final answer within \\boxed{}.\n"
        f"Problem: {question}"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs_list = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW,
                do_sample=True, temperature=TEMP, top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs_list.append(
            tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        )
    return outputs_list

results = {}
for bench_name, rows in benchmarks.items():
    out_path = f"{EVAL_DIR}/{os.path.basename(CHECKPOINT)}_{bench_name}.jsonl"
    if os.path.exists(out_path):
        print(f"[Skip] {out_path} exists.")
        results[bench_name] = out_path
        continue
    with open(out_path, "w") as fout:
        for row in tqdm(rows, desc=bench_name):
            row["outputs"] = generate_answers(row["question"])
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    results[bench_name] = out_path
    print(f"✅ {bench_name} → {out_path}")
```

### Cell 6 — Evaluate (reuse existing verifier)

```python
import signal, re
from math_verify import parse, verify

def extract_boxed(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    return m[-1] if m else ""

def score_outputs(outputs: list[str], answer: str) -> list[int]:
    labels = []
    for pred in outputs:
        try:
            p = parse(extract_boxed(pred))
            g = parse("$" + str(answer) + "$")
            labels.append(int(bool(verify(g, p))))
        except Exception:
            labels.append(0)
    return labels

def compute_metrics(result_file: str) -> dict:
    with open(result_file) as f:
        rows = [json.loads(l) for l in f]
    pass1_list, passn_list, acc_list = [], [], []
    for row in rows:
        labels = score_outputs(row["outputs"], row["answer"])
        pass1_list.append(labels[0])
        passn_list.append(int(any(labels)))
        acc_list.append(sum(labels) / len(labels))
    return {
        "n":        len(rows),
        "pass@1":   sum(pass1_list) / len(pass1_list),
        f"pass@{N_SAMPLES}": sum(passn_list) / len(passn_list),
        f"acc@{N_SAMPLES}":  sum(acc_list)   / len(acc_list),
    }

for bench_name, fpath in results.items():
    m = compute_metrics(fpath)
    print(f"\n{'='*40}")
    print(f"  {bench_name}  —  {os.path.basename(CHECKPOINT)}")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")
```

### Cell 7 — Ablation Table (run after all checkpoints evaluated)

```python
# Manually fill after running notebooks for each variant:
# P-ALIGN baseline → sft_baseline (standard SFT, binary filter, K=1)
# L1              → sft_L1        (K=4, equal-weight above-mean filter)
# L2              → sft_L2        (K=4, advantage-weighted)
# L2 + pf         → sft_L2_pf     (K=4, advantage-weighted + prefix-feedback)

table = {
    "P-ALIGN (baseline, K=1)": {"AIME24 pass@1": "?", "AIME25 pass@1": "?", "AMC12 pass@1": "?"},
    "L1 (K=4, soft filter)":   {"AIME24 pass@1": "?", "AIME25 pass@1": "?", "AMC12 pass@1": "?"},
    "L2 (K=4, adv-weight)":    {"AIME24 pass@1": "?", "AIME25 pass@1": "?", "AMC12 pass@1": "?"},
    "L2 + prefix-feedback":    {"AIME24 pass@1": "?", "AIME25 pass@1": "?", "AMC12 pass@1": "?"},
}

# Print as markdown table
header = ["Method"] + list(list(table.values())[0].keys())
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
for method, vals in table.items():
    row = [method] + [str(v) for v in vals.values()]
    print("| " + " | ".join(row) + " |")
```

---

## Ablation Experiment Design

Run the notebooks in this order to build the comparison table step-by-step:

| Run | Data file | ABLATION var | What it isolates |
|---|---|---|---|
| 0 (baseline) | P-ALIGN binary filter (K=1 correct only, w=1) | `baseline` | Original P-ALIGN |
| 1 | `sft_data_L1.jsonl` | `L1` | Multi-sample (K=4) soft filter vs binary filter |
| 2 | `sft_data_L2.jsonl` | `L2` | Advantage weighting on top of L1 |
| 3 | `sft_data_L2.jsonl` (built with prefix-feedback ON) | `L2_pf` | Prefix-feedback loop on top of L2 |

For run 0, filter `sft_data_L1.jsonl` to keep only the top-1 continuation (best reward per group)
with weight=1.0, which reproduces binary SFT filtering exactly.

---

## Key Parameters Quick Reference

| Parameter | Value | Where used |
|---|---|---|
| K | 4 | Continuations per question |
| temperature | 0.8 | Generation diversity |
| top_p | 0.95 | Generation |
| τ_low | 0.2 | Prefix-feedback trigger |
| T_max | 3 | Max feedback attempts |
| ε | 1e-6 | Advantage denominator |
| clip_c | 3.0 | L2 advantage clip |
| λ_len | 0.0 (start) | Length penalty in reward |
| LoRA r | 16 | Trainable rank |
| lr | 5e-5 | SFT learning rate |
| epochs | 3 | SFT epochs |
| batch | 4 × 8 accum = 32 | Effective batch size |
| β (DPO) | 0.1 | Deferred |

---

## Implementation Checklist

### Phase 0 — Files to Create

- [ ] `src/reward_weighted_data.py` — extract Step 3–6 functions from nb01 into a module
- [ ] `src/weighted_trainer.py` — extract `WeightedSFTDataset`, `WeightedDataCollator`, `WeightedSFTTrainer` from nb02
- [ ] `notebooks/nb01_phase_a_data_building.ipynb`
- [ ] `notebooks/nb02_phase_b_sft.ipynb`
- [ ] `notebooks/nb03_evaluation.ipynb`

### Phase A — Data Building (nb01)

- [ ] Mount Drive, install deps
- [ ] Load `qizheyanger/P-ALIGN` (parse `instruction` → question + prefix via `_Q_RE`) and join to `simplescaling/s1K-1.1` (get `deepseek_thinking_trajectory` + `solution`) → save `prefixes.jsonl`
- [ ] Load Qwen2.5-1.5B-Instruct
- [ ] Implement & test `sample_k_continuations` (verify diversity)
- [ ] Implement & test `compute_rewards` (verify exact-match logic)
- [ ] Implement & test `compute_advantage` (check all-correct / all-wrong edge cases)
- [ ] Implement & test prefix-feedback loop (check extension logic)
- [ ] Run full data loop → save `sft_data_L1.jsonl`, `sft_data_L2.jsonl`, `dpo_pairs.jsonl`
- [ ] Print stats: total/mixed/all-correct/all-wrong/feedback-triggered

### Phase B — Training (nb02)

- [ ] Mount Drive, install TRL + PEFT
- [ ] Implement `WeightedSFTDataset` with correct label masking (user turn → -100)
- [ ] Verify collator passes `weights` through to trainer
- [ ] Verify `WeightedSFTTrainer.compute_loss` produces reasonable loss values
- [ ] Train L1 variant → save checkpoint
- [ ] Train L2 variant → save checkpoint
- [ ] (Optional) Train baseline variant → save checkpoint

### Evaluation (nb03)

- [ ] Load each checkpoint, run inference on AIME24/AIME25/AMC12
- [ ] Verify math_verify + sympy scoring matches `evaluation.py` behavior
- [ ] Fill ablation table
- [ ] Check if L2 > L1 > baseline on AIME24 (expected: yes)
- [ ] Check if prefix-feedback increases coverage on hard problems

---

## DPO Deferred Plan (for after SFT ablation is complete)

Once SFT ablation shows L2 > baseline, add notebook `nb04_phase_b_dpo.ipynb`:

1. Load `dpo_pairs.jsonl` from Drive
2. Load `sft_L2` checkpoint as both `model` and `ref_model`
3. Use TRL `DPOTrainer` with `beta=0.1`, 1 epoch
4. Precompute reference log-probs once (cache to Drive) to avoid holding two models in VRAM
5. Evaluate on same benchmarks and add row to ablation table

VRAM note: DPO on T4 with Qwen2.5-1.5B requires holding base + LoRA simultaneously.
Use `load_in_4bit=True` (bitsandbytes) for the reference model to save ~1.5 GB.

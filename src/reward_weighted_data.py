"""
reward_weighted_data.py

Steps 3-6 of Plan B: Reward-Weighted Prefix-Alignment Distillation.
Importable module used by nb01_phase_a_data_building.ipynb.

Functions:
  build_prompt               — format (question, prefix) into model input
  sample_k_continuations     — Step 3: generate K continuations
  extract_boxed              — extract \\boxed{} content from generation
  verify_answer              — sympy-backed math verifier with SIGALRM timeout
  compute_rewards            — Step 4: binary reward ± length penalty
  compute_advantage          — Step 5: group-relative advantage normalization
  classify_group             — label a reward vector as all_correct/all_wrong/mixed
  sentence_split             — mirrors binary_search.py split_sentences
  build_prefix_ladder        — cumulative sentence-level prefixes from full CoT
  get_next_prefix            — Step 6: next longer prefix in ladder
"""

import re
import math
import signal
from collections import Counter

import torch
from math_verify import parse, verify

# ── Hyperparameter defaults (caller can override) ────────────────────────────
K          = 4
TEMP       = 0.8
TOP_P      = 0.95
MAX_NEW    = 2048
TAU_LOW    = 0.2
T_MAX      = 3
LAMBDA_LEN = 0.0
EPSILON    = 1e-6
CLIP_C     = 3.0


# ── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(question: str, prefix: str) -> str:
    return (
        "Please continue from the draft and solve the problem step by step, "
        "and put your final answer within \\boxed{}. "
        "I will provide you with some prior knowledge as a draft to assist you.\n"
        f"*Question*: {question}\n"
        f"*Prefix*: {prefix}"
    )


# ── Step 3: K-Continuation Sampler ───────────────────────────────────────────

def sample_k_continuations(
    question: str,
    prefix: str,
    model,
    tokenizer,
    k: int = K,
    temp: float = TEMP,
    top_p: float = TOP_P,
    max_new: int = MAX_NEW,
) -> list:
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
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        cont = tokenizer.decode(
            out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        continuations.append(cont)

    answers = [extract_boxed(c) for c in continuations]
    if len(Counter(answers)) == 1:
        print("  [warn] All K continuations produced identical answers — consider raising temperature.")

    return continuations


# ── Step 4: Verifier ─────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1] if matches else ""


def verify_answer(pred: str, gold: str, timeout_sec: int = 5) -> int:
    """Return 1 if pred matches gold, else 0. SIGALRM timeout guards against sympy hangs."""
    def _handler(s, f):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        pred_ans = extract_boxed(pred)
        if not pred_ans:
            return 0
        p = parse(pred_ans)
        g = parse("$" + str(gold) + "$")
        return int(bool(verify(g, p)))
    except Exception:
        return 0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def compute_rewards(
    continuations: list,
    answer: str,
    lambda_len: float = LAMBDA_LEN,
) -> list:
    rewards = []
    for y in continuations:
        r = float(verify_answer(y, answer))
        if lambda_len > 0:
            r -= lambda_len * len(y.split())
        rewards.append(r)
    return rewards


# ── Step 5: Group-Relative Advantage ─────────────────────────────────────────

def compute_advantage(rewards: list, epsilon: float = EPSILON) -> list:
    mu = sum(rewards) / len(rewards)
    variance = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(variance) + epsilon
    return [(r - mu) / sigma for r in rewards]


def classify_group(rewards: list) -> str:
    binary = [1 if r > 0 else 0 for r in rewards]
    if all(b == 1 for b in binary):
        return "all_correct"
    if all(b == 0 for b in binary):
        return "all_wrong"
    return "mixed"


# ── Step 6: Prefix-Feedback Helpers ──────────────────────────────────────────

def sentence_split(text: str) -> list:
    """Mirrors binary_search.py split_sentences."""
    parts = text.split(". ")
    sentences = []
    for s in parts:
        s = s.strip()
        if s:
            sentences.append(s if s.endswith(".") else s + ".")
    return sentences


def build_prefix_ladder(row: dict) -> list:
    """
    Cumulative sentence-level prefixes from the full DeepSeek-R1 CoT
    (row["full_cot"] = deepseek_thinking_trajectory from s1K-1.1).
    Returns ["sent1.", "sent1. sent2.", ..., full_cot].
    """
    full_cot = row.get("full_cot", "")
    sentences = sentence_split(full_cot)
    return [" ".join(sentences[:i + 1]) for i in range(len(sentences))]


def get_next_prefix(current_prefix: str, ladder: list):
    """Return first ladder entry strictly longer than current_prefix, or None."""
    cur_len = len(current_prefix)
    for candidate in ladder:
        if len(candidate) > cur_len:
            return candidate
    return None


# ── Full per-row pipeline (convenience wrapper) ───────────────────────────────

def process_row(
    row: dict,
    model,
    tokenizer,
    k: int = K,
    tau_low: float = TAU_LOW,
    t_max: int = T_MAX,
    clip_c: float = CLIP_C,
    epsilon: float = EPSILON,
    lambda_len: float = LAMBDA_LEN,
) -> dict:
    """
    Run Steps 3-6 for a single prefixes.jsonl row.
    Returns a dict with keys:
      prefix, continuations, rewards, advantages, group_type,
      pass_rate, sft_l1, sft_l2, dpo_pair (may be None)
    """
    question = row["question"]
    prefix   = row["prefix"]
    answer   = str(row["answer"])
    ladder   = build_prefix_ladder(row)

    for _ in range(t_max):
        continuations = sample_k_continuations(question, prefix, model, tokenizer, k=k, lambda_len=lambda_len)
        rewards       = compute_rewards(continuations, answer, lambda_len=lambda_len)
        pass_rate     = sum(1 for r in rewards if r > 0) / k

        if pass_rate < tau_low:
            next_p = get_next_prefix(prefix, ladder)
            if next_p:
                prefix = next_p
                continue
        break

    group_type = classify_group(rewards)
    advantages = compute_advantage(rewards, epsilon=epsilon)

    sft_l1_samples = []
    sft_l2_samples = []
    dpo_pair       = None

    if group_type != "all_wrong":
        mu_r      = sum(rewards) / len(rewards)
        sum_w_l1  = sum(1.0 for r in rewards if r >= mu_r) + epsilon
        sum_w_l2  = sum(max(advantages[i], 0) for i in range(k)) + epsilon

        src_id = str(row.get("id", question[:40]))
        for i, (y, r, a_i) in enumerate(zip(continuations, rewards, advantages)):
            full_seq = prefix + "\n" + y
            w_l1 = (1.0 / sum_w_l1) if r >= mu_r else 0.0
            w_l2 = min(max(a_i, 0), clip_c) / sum_w_l2

            base = {
                "source_id": src_id,
                "question":  question,
                "prefix":    prefix,
                "continuation": y,
                "full_sequence": full_seq,
                "answer":    answer,
                "reward":    r,
                "pass_rate": pass_rate,
            }
            if w_l1 > 0:
                sft_l1_samples.append({**base, "weight": w_l1})
            if w_l2 > 0:
                sft_l2_samples.append({**base, "weight": w_l2})

        if group_type == "mixed":
            best_i  = max(range(k), key=lambda i: rewards[i])
            worst_i = min(range(k), key=lambda i: rewards[i])
            if rewards[best_i] > rewards[worst_i]:
                dpo_pair = {
                    "source_id":       src_id,
                    "question":        question,
                    "prefix":          prefix,
                    "chosen":          continuations[best_i],
                    "rejected":        continuations[worst_i],
                    "reward_chosen":   rewards[best_i],
                    "reward_rejected": rewards[worst_i],
                }

    return {
        "prefix":        prefix,
        "continuations": continuations,
        "rewards":       rewards,
        "advantages":    advantages,
        "group_type":    group_type,
        "pass_rate":     pass_rate,
        "sft_l1":        sft_l1_samples,
        "sft_l2":        sft_l2_samples,
        "dpo_pair":      dpo_pair,
    }

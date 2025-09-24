# peft_full_suite.py
"""
Train **CodeGPT-small-py** (or any GPT‑2‑style checkpoint) on a **secure.jsonl**
file formatted as **ChatML‑style conversations**.  Each line looks like:

```json
{
  "messages": [
    {"role": "user", "content": "<task + template prompt>"},
    {"role": "assistant", "content": "<secure code solution>"}
  ]
}
```

The script supports six adaptation strategies on a single 6 GB GPU:

| Flag      | Method                                             |
|-----------|----------------------------------------------------|
| `full`    | Full‑parameter fine‑tune (FP16)                    |
| `lora`    | Classic LoRA                                       |
| `qlora`   | LoRA on 4‑bit NF4 backbone (the QLoRA recipe)      |
| `adapter` | Residual Houlsby adapters (`adapter‑transformers`) |
| `prefix`  | Prefix‑Tuning (layer‑wise virtual tokens)          |
| `prompt`  | Soft Prompt Tuning (input‑level virtual tokens)    |

Quick examples
--------------
```bash
# LoRA (rank‑8)
python peft_full_suite.py --dataset datasets/original/secure.jsonl --method lora --output_dir lora-secure

# QLoRA
python peft_full_suite.py --dataset secure.jsonl --method qlora --output_dir qlora-secure
```

Dependencies
------------
```bash
pip install "torch>=2.2" "transformers>=4.40" peft>=0.9 \
            bitsandbytes>=0.43.0 accelerate datasets evaluate safetensors \
            # only for --method adapter
            "adapter-transformers[torch]>=3.3"
```


Nota: per lora serve triton==2.1.0
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import bitsandbytes as bnb  # 4/8-bit kernels + paged optimisers
except ImportError:
    bnb = None

'''from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    get_peft_model,
    prepare_model_for_kbit_training,
)'''

try:
    from peft import (
        LoraConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        PromptTuningInit,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


try:
    from transformers import AutoModelWithHeads  # adapter-transformers
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Full & PEFT fine-tuning suite for CodeGPT-small-py")
    p.add_argument("--dataset", required=True, help="Path to secure.jsonl")
    p.add_argument("--method", required=True, choices=[
        "full", "lora", "qlora", "adapter", "prefix", "prompt"],
        help="Choose fine-tuning strategy")

    p.add_argument("--model_id", default="microsoft/CodeGPT-small-py")
    p.add_argument("--output_dir", default="ft-output")

    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, help="Debug: truncate dataset")

    # LoRA specifics
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Prefix / Prompt specifics
    p.add_argument("--num_virtual_tokens", type=int, default=20)

    return p.parse_args()

# ---------------------------------------------------------------------------
# 2. Dataset utils – convert ChatML → {prompt, code}
# ---------------------------------------------------------------------------
def read_jsonl(path: str) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            msgs = raw.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            assistant_msg = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if not user_msg or not assistant_msg:
                continue  # skip malformed entries
            cleaned.append({"prompt": user_msg.strip(), "code": assistant_msg.strip()})
    return cleaned

# ---------------------------------------------------------------------------
# 3. Prompt formatting & tokenisation
# ---------------------------------------------------------------------------
def format_example(ex: Dict[str, str]) -> str:
    return f"PROMPT:\n{ex['prompt']}\n###\nCODE:\n{ex['code']}"

# ---------------------------------------------------------------------------
# 4. Model loader helpers
# ---------------------------------------------------------------------------
def load_backbone(model_id: str, quant_4bit: bool):
    if quant_4bit and bnb is None:
        raise SystemExit("bitsandbytes not installed but --method qlora selected")
    q_cfg = (
        bnb.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        if quant_4bit else None
    )
    return AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=q_cfg
    )

# ---------------------------------------------------------------------------
# 5. Main flow
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 5.1 Load + split dataset
    rows = read_jsonl(args.dataset)
    if args.max_samples:
        rows = rows[: args.max_samples]
    dataset = Dataset.from_list(rows)
    train_raw, eval_raw = dataset.train_test_split(test_size=0.1, seed=args.seed).values()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="right")
    # For GPT2 models, pad_token may be None; set to eos_token to avoid errors.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    '''def tok_fn(example):
        enc = tokenizer(
            format_example(example),
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        # Make sure labels are a list of ints and pad as -100 for ignored tokens
        labels = enc["input_ids"].copy()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        labels = [label if label != pad_token_id else -100 for label in labels]
        enc["labels"] = labels
        return enc'''
    
    def tok_fn(examples):
        inputs = []
        labels = []

        for msgs in examples["messages"]:
            if isinstance(msgs, list) and len(msgs) >= 2:
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

                # Aggiungi EOS alla risposta
                assistant_msg = assistant_msg.strip() + tokenizer.eos_token

                # Concatenazione prompt + risposta
                full_text = user_msg.strip() + "\n" + assistant_msg

                # Tokenizzazione completa (senza doppio tokenizer)
                tokenized = tokenizer(full_text, max_length=512, truncation=True, padding="max_length")
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

                # Calcola la lunghezza tokenizzata del prompt (senza special tokens)
                prompt_len = len(tokenizer(user_msg.strip(), add_special_tokens=False)["input_ids"])

                # Costruisci le label:
                # - Prompt mascherato (token < prompt_len)
                # - Padding mascherato (dove attention_mask == 0)
                # - Codice + EOS appresi
                label_ids = [
                    token if idx >= prompt_len and attn == 1 else -100
                    for idx, (token, attn) in enumerate(zip(input_ids, attention_mask))
                ]

                inputs.append(input_ids)
                labels.append(label_ids)
            else:
                tokenized = tokenizer("", max_length=512, truncation=True, padding="max_length")
                inputs.append(tokenized["input_ids"])
                labels.append([-100] * len(tokenized["input_ids"]))

        return {
            "input_ids": inputs,
            "attention_mask": [tokenizer.get_attention_mask(i, max_length=512) if hasattr(tokenizer, 'get_attention_mask') else [1 if token != tokenizer.pad_token_id else 0 for token in inp] for inp in inputs],
            "labels": labels
        }

    train_ds = train_raw.map(tok_fn, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(tok_fn, remove_columns=eval_raw.column_names)

    # --- DEBUG: print first tokenized example ---
    # print("Sample tokenized train_ds[0]:", train_ds[0])

    # Use default_data_collator for CausalLM/PEFT tasks
    collator = default_data_collator

    # 5.2 Build model according to method
    method = args.method.lower()

    if method in {"lora", "qlora", "prefix", "prompt"} and not PEFT_AVAILABLE:
        raise SystemExit("PEFT not installed → pip install 'peft>=0.9'")

    if method == "full":
        model = load_backbone(args.model_id, quant_4bit=False)
    elif method in {"lora", "qlora", "prefix", "prompt"}:
        quant = method == "qlora"
        base = load_backbone(args.model_id, quant_4bit=quant)
        base = prepare_model_for_kbit_training(base) if quant else base
        if method in {"lora", "qlora"}:
            cfg = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                task_type="CAUSAL_LM",
                bias="none",
                target_modules=["c_attn", "c_proj"],
            )
        elif method == "prefix":
            cfg = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=args.num_virtual_tokens)
        else:  # prompt
            cfg = PromptTuningConfig(
                task_type="CAUSAL_LM",
                num_virtual_tokens=args.num_virtual_tokens,
                prompt_tuning_init=PromptTuningInit.RANDOM,
            )
        model = get_peft_model(base, cfg)
    elif method == "adapter":
        if not ADAPTERS_AVAILABLE:
            raise SystemExit("adapter-transformers not installed → pip install 'adapter-transformers[torch]>=3.3'")
        model = AutoModelWithHeads.from_pretrained(args.model_id, device_map="auto")
        model.add_adapter("houlsby", config="houlsby")
        model.add_causal_lm_head("houlsby")  # aggiungi la head per causal LM
        model.set_active_adapters("houlsby")
        model.train_adapter(["houlsby"])
    else:
        raise ValueError("Unknown method")

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        # For non-PEFT models: print standard trainable param count
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100 * trainable / total
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")


    # 5.3 TrainingArguments
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        # gradient_checkpointing=True,   # Disable until confirmed working
        optim="paged_adamw_8bit" if (method == "qlora" and bnb is not None) else "adamw_torch",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=8,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()

    # 5.4 Save checkpoints / adapters
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if method == "lora":
        model.merge_and_unload().save_pretrained(args.output_dir, safe_serialization=True)
    elif method == "adapter":
        model.save_adapter(args.output_dir, "houlsby")
    else:
        model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✔ Training finished → weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()

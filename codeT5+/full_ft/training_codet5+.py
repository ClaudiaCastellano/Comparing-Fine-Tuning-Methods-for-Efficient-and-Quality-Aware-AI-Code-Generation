# Training CodeT5+ (encoder–decoder) on "user(content=prompt+partial)" -> "assistant(content=complete code)"
# Dataset: ogni esempio ha "messages": [{"role":"user","content":...}, {"role":"assistant","content":...}]

import argparse
import json
import os
import time
from typing import Dict, List, Union

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, required=True, help="Path to a JSON/JSONL file with training data.")
    p.add_argument("--eval_file", type=str, default=None, help="Optional JSON/JSONL evaluation file.")
    p.add_argument("--model_name", type=str, default="Salesforce/codet5p-220m", help="HF model id or local path.")
    p.add_argument("--output_dir", type=str, default="./codet5p-out-full-ft")

    # Lunghezze input/target
    p.add_argument("--max_source_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=512)

    # Iperparametri
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=float, default=5.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=42)

    # Mixed precision
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    # Logging & checkpointing
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=None, help="Usato solo se eval_strategy=steps")
    p.add_argument("--save_steps", type=int, default=None, help="Usato solo se save_strategy=steps")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--load_best_model_at_end", action="store_true")
    p.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    p.add_argument("--greater_is_better", action="store_true")

    # Hub & memoria
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Generazione in eval
    p.add_argument("--num_beams_eval", type=int, default=1, help="Beams for evaluation generation.")

    # Map settings
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--disable_cache", action="store_true", help="datasets.map(load_from_cache_file=False)")

    return p.parse_args()

def build_datasets(train_file: str, eval_file: Union[str, None]) -> DatasetDict:
    ddict = {}
    ddict["train"] = load_dataset("json", data_files=train_file, split="train")
    if eval_file:
        ddict["validation"] = load_dataset("json", data_files=eval_file, split="train")
    return DatasetDict(ddict)

def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    args = parse_args()
    set_seed(args.seed)

    # Tokenizer & modello
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Ensure pad/eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Info parametri
    total_params, trainable_params = count_parameters(model)
    print(f"🧠 Parametri totali: {total_params:,}")
    print(f"🧠 Parametri allenabili: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Datasets
    dsd = build_datasets(args.train_file, args.eval_file)

    def preprocess(example):
        # Assumi: example["messages"] = [
        #   {"role": "user", "content": "<prompt + codice parziale>"},
        #   {"role": "assistant", "content": "<codice corretto>"}
        # ]
        user_text = example["messages"][0]["content"]
        target_text = example["messages"][1]["content"]

        model_inputs = tokenizer(
            user_text,
            max_length=args.max_source_length,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            target_text.strip() + (tokenizer.eos_token or ""),
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )["input_ids"]

        model_inputs["labels"] = labels
        return model_inputs


    # Map + cache
    dsd_proc = DatasetDict()
    for split_name, ds in dsd.items():
        dsd_proc[split_name] = ds.map(
            preprocess,
            batched=False,
            num_proc=args.num_proc,
            remove_columns=ds.column_names,
            load_from_cache_file=not args.disable_cache,
            desc=f"Tokenizing {split_name}",
        )

    # Collator
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,

        eval_strategy=args.eval_strategy if "validation" in dsd_proc else "no",
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,

        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,

        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams_eval,

        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,

        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,

        report_to=["none"],
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,

        remove_unused_columns=False,  # mantieni le colonne tokenizzate
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dsd_proc["train"],
        eval_dataset=dsd_proc.get("validation"),
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Monitoraggio
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    trainer.train()

    end_time = time.time()
    training_time = end_time - start_time
    max_memory = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0  # GB

    # Salvataggio finale
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    # Calcolo dimensione modello salvato
    def get_model_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
        return total_size / (1024 ** 2)  # MB

    model_size_mb = get_model_size(args.output_dir)

    # Report risorse
    log_info = {
        "technique": "full_finetuning_seq2seq_user_to_assistant",
        "training_time_sec": training_time,
        "gpu_max_memory_GB": max_memory,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "model_disk_size_MB": model_size_mb,
    }

    with open(os.path.join(args.output_dir, "resource_report.json"), "w") as f:
        json.dump(log_info, f, indent=4)

    print("✅ Report salvato in resource_report.json")

if __name__ == "__main__":
    main()

# Training CodeT5+ with LoRA / QLoRA on "prompt + partial code" -> "complete code"
# I/O identico al tuo script: input JSON/JSONL con un campo "messages" (array di turni).

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

# Opzionali per 8/4-bit
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# PEFT (LoRA / QLoRA)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, required=True, help="Path a JSON/JSONL con i dati di training.")
    p.add_argument("--eval_file", type=str, default=None, help="Path a JSON/JSONL di validation (opzionale).")
    p.add_argument("--model_name", type=str, default="Salesforce/codet5p-220m", help="HF model id o path locale.")
    p.add_argument("--output_dir", type=str, default="./codet5p-out-lora")

    # Lunghezze
    p.add_argument("--max_source_length", type=int, default=1024)
    p.add_argument("--max_target_length", type=int, default=512)

    # Iperparametri
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-4)  # tipico per LoRA
    p.add_argument("--num_train_epochs", type=float, default=5.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=42)

    # Precisione & memoria
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_8bit", action="store_true", help="Carica il modello in 8-bit (QLoRA).")
    p.add_argument("--use_4bit", action="store_true", help="Carica il modello in 4-bit (QLoRA).")

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o",
        help="Lista separata da virgole dei moduli target (T5 usa tipicamente q,k,v,o).",
    )
    p.add_argument("--merge_and_unload", action="store_true",
                   help="Tenta il merge delle LoRA nel backbone a fine training (sconsigliato con 4/8-bit).")

    # Logging & ckpt
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--load_best_model_at_end", action="store_true")
    p.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    p.add_argument("--greater_is_better", action="store_true")

    # Hub
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)

    # Generazione in eval
    p.add_argument("--num_beams_eval", type=int, default=1)

    # Map settings
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--disable_cache", action="store_true")

    return p.parse_args()


def linearize_messages(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def extract_target_from_messages(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


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


def parse_target_modules(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    args = parse_args()
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token

    # Modello (eventuale quantizzazione per QLoRA)
    quant_config = None
    device_map = None
    if (args.use_4bit or args.use_8bit):
        if not _HAS_BNB:
            raise RuntimeError("BitsAndBytesConfig non trovato. Installa: pip install bitsandbytes accelerate")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            load_in_8bit=(not args.use_4bit) and args.use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map=device_map,
    )

    # Assicura pad/eos id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Prepara per k-bit training se quantizzato
    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Configura LoRA
    target_modules = parse_target_modules(args.lora_target_modules)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="SEQ_2_SEQ_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    # Stampa conteggio parametri
    total_params, trainable_params = count_parameters(model)
    print(f"🧠 Parametri totali: {total_params:,}")
    print(f"🧠 Parametri allenabili: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

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

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

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

        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dsd_proc["train"],
        eval_dataset=dsd_proc.get("validation"),
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Monitoraggio risorse
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    trainer.train()

    end_time = time.time()
    training_time = end_time - start_time
    max_memory = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0  # GB

    # Salvataggio: per PEFT, save_pretrained salva SOLO gli adapter LoRA
    print("💾 Salvataggio adapter LoRA...")
    #trainer.save_model()  # chiama model.save_pretrained(output_dir) su PeftModel => salva adapter
    model.save_pretrained(args.output_dir)  
    tokenizer.save_pretrained(args.output_dir)

    # Opzionale: merge delle LoRA nel backbone (sconsigliato in 4/8-bit)
    if args.merge_and_unload:
        try:
            print("🔁 Merge LoRA nel modello base e salvataggio...")
            merged = model.merge_and_unload()
            merged.save_pretrained(os.path.join(args.output_dir, "merged"))
        except Exception as e:
            print(f"Merge non eseguito: {e}")

    # Report dimensioni/cartella (adapter)
    def get_dir_size_mb(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total / (1024 ** 2)

    model_size_mb = get_dir_size_mb(args.output_dir)

    log_info = {
        "technique": "lora_or_qlora_seq2seq",
        "training_time_sec": training_time,
        "gpu_max_memory_GB": max_memory,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "model_disk_size_MB": model_size_mb,
        #"quantized": bool(quant_config is not None),
        #"use_4bit": bool(quant_config.load_in_4bit) if quant_config else False,
        #"use_8bit": bool(quant_config.load_in_8bit) if quant_config else False,
        #"lora_r": args.lora_r,
        #"lora_alpha": args.lora_alpha,
        #"lora_dropout": args.lora_dropout,
        #"lora_target_modules": target_modules,
    }

    with open(os.path.join(args.output_dir, "resource_report.json"), "w") as f:
        json.dump(log_info, f, indent=4)

    print("✅ Report salvato in resource_report.json")


if __name__ == "__main__":
    main()

import torch
import os
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Configurazioni base
model_name = "microsoft/CodeGPT-small-py"
max_source_length = 512
batch_size = 4
num_train_epochs = 5
grad_accumulation_steps = 8
learning_rate = 5e-5
output_dir = "./codegpt_full_ft_with_performance_eostoken_32batch"

# 2. Carica tokenizer e modello
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"  # Assicura padding a destra
model = AutoModelForCausalLM.from_pretrained(model_name)
# Imposta eos_token e pad_token se non esistono
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # GPT-style
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


# 2b. Conta i parametri allenabili e totali
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(model)
print(f"🧠 Parametri totali: {total_params:,}")
print(f"🧠 Parametri allenabili: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# 3. Funzione di preprocessing
def preprocess_function(examples):
    inputs = []
    labels = []

    for msgs in examples["messages"]:
        if isinstance(msgs, list) and len(msgs) >= 2:
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

            # Aggiunge EOS alla risposta
            assistant_msg = assistant_msg.strip() + tokenizer.eos_token

            # Concatenazione prompt + risposta
            full_text = user_msg.strip() + "\n" + assistant_msg

            # Tokenizzazione completa (senza doppio tokenizer)
            tokenized = tokenizer(full_text, max_length=max_source_length, truncation=True, padding="max_length")
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
            tokenized = tokenizer("", max_length=max_source_length, truncation=True, padding="max_length")
            inputs.append(tokenized["input_ids"])
            labels.append([-100] * len(tokenized["input_ids"]))

    return {
        "input_ids": inputs,
        "attention_mask": [tokenizer.get_attention_mask(inp, max_length=max_source_length) if hasattr(tokenizer, 'get_attention_mask') else [1 if token != tokenizer.pad_token_id else 0 for token in inp] for inp in inputs],
        "labels": labels
    }


# 4. Carica dataset
dataset = load_dataset("json", data_files={
    "train": "../datasets/train_datasets/train_secure.jsonl",
    "validation": "../datasets/validation_datasets/validation_secure.jsonl",
})

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# 5. Tokenizza dataset
tokenized_train = train_dataset.map(
    preprocess_function, 
    batched=True,
    num_proc=32, 
    remove_columns=train_dataset.column_names,
    load_from_cache_file=False
)

tokenized_validation = val_dataset.map(
    preprocess_function, 
    batched=True,
    num_proc=32, 
    remove_columns=val_dataset.column_names,
    load_from_cache_file=False
)

print("📊 Train set size:", len(tokenized_train))
print("📊 Validation set size:", len(tokenized_validation))

# 6. Argomenti di training
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accumulation_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    save_total_limit=2,
    fp16=True,
    logging_dir="./logs",
    logging_steps=20,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer
)

# 8. MONITORAGGIO: tempo, memoria, training
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

trainer.train()

end_time = time.time()
training_time = end_time - start_time
max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB

print(f"⏱️ Tempo di training: {training_time:.2f} secondi")
print(f"📈 VRAM massima usata: {max_memory:.2f} GB")

# 9. Dimensione modello salvato
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

def get_model_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 ** 2)  # MB

model_size = get_model_size(output_dir)
print(f"💾 Spazio disco occupato dal modello fine-tuned: {model_size:.2f} MB")

# 10. Log finale JSON (utile per confronto tra tecniche)
log_info = {
    "technique": "full_finetuning",
    "training_time_sec": training_time,
    "gpu_max_memory_GB": max_memory,
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "model_disk_size_MB": model_size
}

with open(os.path.join(output_dir, "resource_report.json"), "w") as f:
    json.dump(log_info, f, indent=4)

print("✅ Report salvato in resource_report.json")

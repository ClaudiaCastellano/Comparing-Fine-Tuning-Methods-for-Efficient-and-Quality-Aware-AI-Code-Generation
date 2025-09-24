import torch
import os
import time
import json
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM
from transformers.adapters import AdapterConfig
from transformers import AutoTokenizer, AutoModelWithHeads

# 1. Configurazioni base
model_name = "microsoft/CodeGPT-small-py"
max_source_length = 512
batch_size = 4
num_train_epochs = 5
grad_accumulation_steps = 8
learning_rate = 5e-4
output_dir = "./codegpt_adapters_ft"

# 2. Carica tokenizer e modello con supporto adapter
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# 2b. Aggiungi un adapter standard (Houlsby)
adapter_name = "code_adapter"
adapter_config = AdapterConfig.load("houlsby")
model.add_adapter(adapter_name, config=adapter_config)
model.train_adapter(adapter_name)
model.set_active_adapters(adapter_name)


# 2c. Conta i parametri
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(model)
print(f"🧠 Parametri totali: {total_params:,}")
print(f"🧠 Parametri allenabili: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# 3. Preprocessing
def preprocess_function(examples):
    inputs, labels, attention_masks = [], [], []

    for msgs in examples["messages"]:
        if isinstance(msgs, list) and len(msgs) >= 2:
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            assistant_msg = assistant_msg.strip() + tokenizer.eos_token
            full_text = user_msg.strip() + "\n" + assistant_msg

            tokenized = tokenizer(full_text, max_length=max_source_length, truncation=True, padding="max_length")
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            prompt_len = len(tokenizer(user_msg.strip(), add_special_tokens=False)["input_ids"])

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

        attention_masks = [
            [1 if token != tokenizer.pad_token_id else 0 for token in input_ids] for input_ids in inputs
        ]
    return {
        "input_ids": inputs,
        "attention_mask": attention_masks,
        "labels": labels
    }


# 4. Caricamento dataset
dataset = load_dataset("json", data_files={
    "train": "../datasets/train_datasets/train_secure.jsonl",
    "validation": "../datasets/validation_datasets/validation_secure.jsonl",
}, cache_dir="./dataset_cache", verification_mode="no_checks")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]


# 5. Tokenizzazione
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=16,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=False
)

tokenized_validation = val_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=16,
    remove_columns=val_dataset.column_names,
    load_from_cache_file=False
)

print("📊 Train set size:", len(tokenized_train))
print("📊 Validation set size:", len(tokenized_validation))

# 6. Argomenti di training
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
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

# 8. MONITORAGGIO
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

trainer.train()

end_time = time.time()
training_time = end_time - start_time
max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

print(f"⏱️ Tempo di training: {training_time:.2f} secondi")
print(f"📈 VRAM massima usata: {max_memory:.2f} GB")

# 9. Salvataggio dell'adapter
model.save_adapter(output_dir, adapter_name)
tokenizer.save_pretrained(output_dir)

def get_model_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 ** 2)

model_size = get_model_size(output_dir)
print(f"💾 Spazio disco occupato dall'adapter: {model_size:.2f} MB")

# 10. Report finale
log_info = {
    "technique": "adapter_houlsby",
    "training_time_sec": training_time,
    "gpu_max_memory_GB": max_memory,
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "model_disk_size_MB": model_size
}

with open(os.path.join(output_dir, "resource_report.json"), "w") as f:
    json.dump(log_info, f, indent=4)

print("✅ Report salvato in resource_report.json")

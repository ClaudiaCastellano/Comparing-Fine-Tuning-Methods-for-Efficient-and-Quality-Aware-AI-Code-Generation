import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig

# Funzione per caricare il dataset di test
def load_test_data(path):
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset

# Funzione per estrazione del prompt
def extract_prompt(example):
    user_msg = next((msg["content"] for msg in example["messages"] if msg["role"] == "user"), "").strip()
    return {"prompt": user_msg + "\n"}

# Funzione per generazione del codice
def generate_code(example, tokenizer, model, device):
    input_ids = tokenizer(
        example["prompt"], return_tensors="pt", truncation=True, max_length=512
    ).input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )

    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return {"generated_code": generated_text}

def main(args):
    print("📥 Loading dataset...")
    dataset = load_test_data(args.input_file)

    print("✂️ Extracting prompts...")
    dataset = dataset.map(extract_prompt)

    print(f"🧠 Loading tokenizer and Prefix Tuning model from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    # Caricamento configurazione PEFT
    peft_config = PeftConfig.from_pretrained(args.model_name)

    # Caricamento modello base
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)

    # Applicazione Prefix Adapter
    model = PeftModel.from_pretrained(base_model, args.model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("⚙️ Running inference...")
    generated_outputs = []
    for example in tqdm(dataset):
        result = generate_code(example, tokenizer, model, device)
        reference = next((m["content"] for m in example["messages"] if m["role"] == "assistant"), None)
        result_row = {
            "prompt": example["prompt"],
            "generated_code": result["generated_code"],
            "reference": reference
        }
        generated_outputs.append(result_row)

    print(f"💾 Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in generated_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with CodeGPT + Prefix Tuning on a JSONL test set.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output .jsonl")
    parser.add_argument("--model_name", type=str, required=True, help="Path to directory with PEFT prefix-tuned model")

    args = parser.parse_args()
    main(args)

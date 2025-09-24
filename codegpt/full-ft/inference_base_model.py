import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Funzione per caricamento test set
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

    output_ids = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )


    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return {"generated_code": generated_text}

def main(args):
    print("📥 Loading dataset...")
    dataset = load_test_data(args.input_file)

    print("✂️ Extracting prompts...")
    dataset = dataset.map(extract_prompt)

    print(f"🧠 Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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
    parser = argparse.ArgumentParser(description="Run inference with CodeGPT on a JSONL test set.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output .jsonl")
    parser.add_argument("--model_name", type=str, default="microsoft/CodeGPT-small-py", help="Model name or path")

    args = parser.parse_args()
    main(args)

# sh -c "python inference_base_model.py --input_file ../datasets/split/secure/secure_test.jsonl --output_file prova_output.jsonl --model_name microsoft/CodeGPT-small-py"
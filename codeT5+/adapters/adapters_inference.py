import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

def load_test_data(path):
    return load_dataset("json", data_files=path, split="train")

def extract_prompt_two_turns(example, strict=False):
    """
    Usa SOLO il content dell'USER (turno 0) come prompt.
    Non include mai il content dell'assistant: niente leak.
    """
    msgs = example.get("messages", [])
    if strict:
        assert isinstance(msgs, list) and len(msgs) == 2, "Ogni esempio deve avere esattamente 2 turni"
        assert msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant", "Ordine ruoli atteso: user, assistant"

    user_text = ""
    if isinstance(msgs, list) and len(msgs) >= 1:
        if msgs[0].get("role") == "user":
            user_text = msgs[0].get("content", "")
        else:
            user_text = next((m.get("content","") for m in msgs if m.get("role") == "user"), "")
    return {"prompt": (user_text or "").strip()}

def generate_code(example, tokenizer, model, device, max_source_length=512, max_new_tokens=256, num_beams=1, do_sample=False, temperature=0.7, top_p=0.9, top_k=0):
    inputs = tokenizer(
        example["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
        if top_k and top_k > 0:
            gen_kwargs.update(dict(top_k=top_k))

    output_ids = model.generate(**inputs, **gen_kwargs)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_code": generated_text}

def main(args):
    print("📥 Loading dataset...")
    dataset = load_test_data(args.input_file)

    print("✂️ Extracting prompts (2-turns, no-leak)...")
    dataset = dataset.map(lambda ex: extract_prompt_two_turns(ex, strict=args.strict))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"🧠 Loading tokenizer and model from checkpoint: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<pad>'
    if tokenizer.eos_token is None:
        tokenizer.eos_token = '</s>'

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.set_active_adapters("code_adapter") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("⚙️ Running inference...")
    generated_outputs = []
    for example in tqdm(dataset):
        result = generate_code(
            example, tokenizer, model, device,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        # riferimento: assistant del secondo turno (mai usato nel prompt)
        reference = None
        msgs = example.get("messages", [])
        if isinstance(msgs, list) and len(msgs) >= 2 and msgs[1].get("role") == "assistant":
            reference = msgs[1].get("content")

        generated_outputs.append({
            "prompt": example["prompt"],
            "generated_code": result["generated_code"],
            "reference": reference
        })

    print(f"💾 Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in generated_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference CodeT5+ (2-turns dataset, no-leak) - full FT o LoRA.")
    parser.add_argument("--input_file", type=str, required=True, help="Path a input .json/.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Path per output .jsonl")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Percorso del modello full oppure della cartella adapter LoRA (contiene adapter_config.json)")

    # Generazione
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)

    # Guardie
    parser.add_argument("--strict", action="store_true", help="Verifica che ogni esempio sia esattamente [user, assistant]")

    args = parser.parse_args()
    main(args)

from __future__ import annotations
import os, argparse, logging, pickle, html as htmlmod, json
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
import shap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# PEFT opzionale
try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("shap_token_dual_source")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
shap.initjs()

MODEL_HUB_ID = {
    "codegpt": "microsoft/CodeGPT-small-py",
    "codet5p": "Salesforce/codet5p-220m",
}

def _is_effectively_empty_text(s: Optional[str]) -> bool:
    """True se la stringa è None, vuota o solo spazi/a-capo."""
    if s is None:
        return True
    if s == "":
        return True
    # se sono solo whitespace, SHAP produrrebbe 0 token
    return s.strip() == ""


# ---------- SAFE LIMITS (fix OverflowError) ----------
def _safe_model_ctx(model, tokenizer, fallback: int = 2048) -> int:
    """Ritorna un contesto massimo ragionevole per il modello."""
    ctx = getattr(model.config, "n_positions", None) \
          or getattr(model.config, "max_position_embeddings", None) \
          or getattr(tokenizer, "model_max_length", None) \
          or fallback
    try:
        ctx = int(ctx)
    except Exception:
        ctx = fallback
    if ctx > 1_000_000:  
        ctx = fallback
    return max(32, ctx)

def _safe_tok_maxlen(tokenizer, fallback: int = 4096) -> int:
    """Cap sicuro per max_length del tokenizer (evita sentinelle enormi)."""
    m = getattr(tokenizer, "model_max_length", None)
    try:
        m = int(m)
    except Exception:
        m = None
    if m is None or m > 1_000_000:
        return fallback
    return max(32, m)

# -------------------- dataset helpers --------------------
def _content_from_msg(msg):
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "\n".join(parts)
    return str(c)

def record_to_last_user_text(rec) -> str:
    msgs = rec.get("messages", [])
    if not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        if (m.get("role","").lower() == "user"):
            return _content_from_msg(m).strip()
    return ""

# -------------------- caricamento modello (full_ft / PEFT / adapters) --------------------
def _set_family_flags(model, family, tokenizer):
    if family == "codet5p":
        if getattr(model.config, "decoder_start_token_id", None) is None:
            model.config.decoder_start_token_id = tokenizer.pad_token_id
    return model

def load_model_any_technique(
    family: str,
    technique: str,
    checkpoint_path: str,
    base_model_hub_id: str,
    torch_dtype,
    device,
    peft_merge: bool = False,
):
    # FULL FT
    if technique == "full_ft":
        load_src = checkpoint_path or base_model_hub_id
        tokenizer = AutoTokenizer.from_pretrained(load_src, use_fast=True)
        if family == "codegpt":
            model = AutoModelForCausalLM.from_pretrained(load_src, torch_dtype=torch_dtype)
        elif family == "codet5p":
            model = AutoModelForSeq2SeqLM.from_pretrained(load_src, torch_dtype=torch_dtype)
        else:
            raise ValueError(f"Famiglia non supportata: {family}")
        model = _set_family_flags(model, family, tokenizer)
        model.to(device).eval()
        return model, tokenizer

    # BASE per PEFT/Adapters
    base_id = base_model_hub_id or MODEL_HUB_ID[family]
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if family == "codegpt":
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch_dtype)
    elif family == "codet5p":
        base = AutoModelForSeq2SeqLM.from_pretrained(base_id, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Famiglia non supportata: {family}")

    # PEFT
    if technique in {"lora", "prefix_tuning", "prompt_tuning"}:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("Richiesto PEFT ma 'peft' non è installato (pip install peft).")
        model = PeftModel.from_pretrained(base, checkpoint_path)
        if peft_merge:
            model = model.merge_and_unload()
        model = _set_family_flags(model, family, tokenizer)
        model.to(device).eval()
        return model, tokenizer

    # Adapters (AdapterHub)
    if technique == "adapters":
        if not (hasattr(base, "load_adapter") and hasattr(base, "set_active_adapters")):
            raise RuntimeError(
                "Gli AdapterHub non sono disponibili nel tuo 'transformers'. "
                "Installa: pip install adapter-transformers"
            )
        adapter_name = "my_adapter"
        base.load_adapter(checkpoint_path, load_as=adapter_name)
        base.set_active_adapters(adapter_name)
        model = _set_family_flags(base, family, tokenizer)
        model.to(device).eval()
        return model, tokenizer

    raise ValueError(f"Tecnica non riconosciuta: {technique}")

# -------------------- generazione output --------------------
@torch.no_grad()
def generate_output_tokens(
    model, tokenizer, device, family: str, text: str, gen_max_new_tokens: int
) -> Tuple[List[int], str]:
    enc = tokenizer([text], return_tensors="pt", padding=False, truncation=False).to(device)
    gen_kwargs = dict(max_new_tokens=gen_max_new_tokens, do_sample=False, num_beams=1, early_stopping=True)

    if family == "codegpt":
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            pad_token_id=pad_id,
            **gen_kwargs
        )  # [1, L_total]
        in_len = int(enc["attention_mask"][0].sum().item()) if "attention_mask" in enc else enc["input_ids"].shape[1]
        gen_ids = out[0, in_len:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return gen_ids, gen_text

    else:  # codet5p
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            **gen_kwargs
        )  # [1, L_out]
        gen_ids = out[0].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return gen_ids, gen_text

# -------------------- predict_fn: SORGENTE = INPUT --------------------
def make_predict_fn_input_source(
    model,
    tokenizer,
    device,
    family: str,
    base_text: str,
    y_ids: List[int],
    t_index: int,
    max_input_len: Optional[int] = None,
):
    """Feature = token dell'input; y_<t> resta fisso."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    if family == "codet5p":
        # decoder: [<start>] + y_<t>
        dec_start = getattr(model.config, "decoder_start_token_id", None) or pad_id
        y_prefix = y_ids[:t_index]
        target_id = y_ids[t_index]
        max_len_enc = _safe_tok_maxlen(tokenizer, fallback=2048)  # FIX

        @torch.no_grad()
        def predict(batch_texts: List[str]) -> np.ndarray:
            enc = tokenizer(
                list(map(str, batch_texts)),
                return_tensors="pt",
                padding=True,
                truncation=True,              # OK
                max_length=max_len_enc        # FIX: cap sicuro
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            B = enc["input_ids"].size(0)
            dec_in = torch.tensor([[dec_start] + y_prefix], dtype=torch.long, device=device).repeat(B, 1)
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask", None),
                        decoder_input_ids=dec_in)
            logits = out.logits[:, -1, :]  # [B, V] predice y_t
            logp = torch.log_softmax(logits, dim=-1)
            scores = logp[:, target_id]
            return scores.detach().cpu().numpy()

        return predict

    else:  # codegpt (causal)
        y_prefix = y_ids[:t_index]
        target_id = y_ids[t_index]
        max_ctx = _safe_model_ctx(model, tokenizer, fallback=2048)  # FIX
        if max_input_len is None:
            max_input_len = max(16, max_ctx - (len(y_prefix) + 8))

        @torch.no_grad()
        def predict(batch_texts: List[str]) -> np.ndarray:
            encoded_seqs = []
            for txt in batch_texts:
                enc = tokenizer(str(txt), return_tensors="pt", padding=False,
                                truncation=True, max_length=max_input_len)  # OK: max_input_len è sicuro
                inp_ids = enc["input_ids"][0]
                if len(y_prefix) > 0:
                    ypref = torch.tensor(y_prefix, dtype=torch.long)
                    inp_ids = torch.cat([inp_ids, ypref], dim=0)
                encoded_seqs.append(inp_ids)

            maxL = max(seq.size(0) for seq in encoded_seqs)
            batch_ids = torch.full((len(encoded_seqs), maxL), pad_id, dtype=torch.long)
            attn = torch.zeros((len(encoded_seqs), maxL), dtype=torch.long)
            for i, seq in enumerate(encoded_seqs):
                L = seq.size(0); batch_ids[i, :L] = seq; attn[i, :L] = 1

            batch_ids = batch_ids.to(device); attn = attn.to(device)
            logits = model(input_ids=batch_ids, attention_mask=attn).logits  # [B, L, V]
            last = logits[:, -1, :]
            logp = torch.log_softmax(last, dim=-1)
            scores = logp[:, target_id]
            return scores.detach().cpu().numpy()

        return predict

# -------------------- predict_fn: SORGENTE = PREFISSO OUTPUT --------------------
def make_predict_fn_prefix_source(
    model,
    tokenizer,
    device,
    family: str,
    fixed_input_text: str,
    y_ids: List[int],
    t_index: int,
    max_input_len: Optional[int] = None,
):

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    y_prefix_ids = y_ids[:t_index]
    target_id = y_ids[t_index]

    if family == "codet5p":
        dec_start = getattr(model.config, "decoder_start_token_id", None) or pad_id

        # Esclude il decoder_start dal prefisso osservato 
        eff_ids = list(y_prefix_ids)
        if len(eff_ids) > 0 and eff_ids[0] == dec_start:
            eff_ids = eff_ids[1:]
        # rimuovi eventuali pad iniziali
        if len(eff_ids) > 0 and tokenizer.pad_token_id is not None and eff_ids[0] == tokenizer.pad_token_id:
            eff_ids = eff_ids[1:]

        prefix_text_base = tokenizer.decode(
            eff_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Se il prefisso "testuale" è vuoto/solo spazi, non ci sono feature da spiegare -> placeholder
        if _is_effectively_empty_text(prefix_text_base):
            return None, ""  

        max_len_enc = _safe_tok_maxlen(tokenizer, fallback=2048)
        max_len_dec = _safe_tok_maxlen(tokenizer, fallback=1024)

        @torch.no_grad()
        def predict(batch_prefix_texts: List[str]) -> np.ndarray:
            # encoder input fisso 
            enc1 = tokenizer([fixed_input_text], return_tensors="pt",
                             padding=False, truncation=True, max_length=max_len_enc).to(device)
            B = len(batch_prefix_texts)

            # tokenizza ciascun prefisso proposto da SHAP
            dec_in_list = []
            for s in batch_prefix_texts:
                if s is None:
                    s = ""
                dec_ids = tokenizer(str(s), return_tensors="pt", padding=False,
                                    truncation=True, max_length=max_len_dec,
                                    add_special_tokens=False)["input_ids"][0].to(device)
                di = torch.cat([torch.tensor([dec_start], device=device), dec_ids], dim=0)
                dec_in_list.append(di)

            maxL = max(x.size(0) for x in dec_in_list) if dec_in_list else 1
            dec_batch = torch.full((B, maxL), pad_id, dtype=torch.long, device=device)
            for i, di in enumerate(dec_in_list):
                dec_batch[i, :di.size(0)] = di

            out = model(
                input_ids=enc1["input_ids"].repeat(B, 1),
                attention_mask=enc1.get("attention_mask", None).repeat(B, 1) if "attention_mask" in enc1 else None,
                decoder_input_ids=dec_batch
            )
            logits = out.logits[:, -1, :]  # [B, V]
            logp = torch.log_softmax(logits, dim=-1)
            scores = logp[:, target_id]
            return scores.detach().cpu().numpy()

        return predict, prefix_text_base

    else:  
        max_ctx = _safe_model_ctx(model, tokenizer, fallback=2048)
        margin = 4

        prefix_text_base = tokenizer.decode(
            y_prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if _is_effectively_empty_text(prefix_text_base):
            return None, "" 

        fixed_enc = tokenizer(str(fixed_input_text), return_tensors="pt",
                              padding=False, truncation=False)
        fixed_ids = fixed_enc["input_ids"][0]
        max_fixed_len = max_ctx - margin
        if fixed_ids.size(0) > max_fixed_len:
            logger.warning(f"[prefix-source/gpt] input fisso clippato {fixed_ids.size(0)}→{max_fixed_len} (ctx={max_ctx})")
            fixed_ids = fixed_ids[-max_fixed_len:]

        @torch.no_grad()
        def predict(batch_prefix_texts: List[str]) -> np.ndarray:
            seqs = []
            for s in batch_prefix_texts:
                if s is None:
                    s = ""
        
                pref_ids = tokenizer(str(s), return_tensors="pt",
                                     padding=False, truncation=False,
                                     add_special_tokens=False)["input_ids"][0]
                
                max_pref_len = max(1, max_ctx - fixed_ids.size(0) - margin)
                if pref_ids.size(0) > max_pref_len:
                    pref_ids = pref_ids[-max_pref_len:]
                seqs.append(torch.cat([fixed_ids, pref_ids], dim=0))

            pad = pad_id
            maxL = max(z.size(0) for z in seqs)
            batch_ids = torch.full((len(seqs), maxL), pad, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxL), dtype=torch.long)
            for i, z in enumerate(seqs):
                L = z.size(0); batch_ids[i, :L] = z; attn[i, :L] = 1

            batch_ids = batch_ids.to(device); attn = attn.to(device)
            logits = model(input_ids=batch_ids, attention_mask=attn).logits  
            last = logits[:, -1, :]                                          
            logp = torch.log_softmax(last, dim=-1)
            scores = logp[:, target_id]
            return scores.detach().cpu().numpy()

        return predict, prefix_text_base


# -------------------- salvataggi: plot per token (HTML) ----------
def save_text_plot_full(expl, path_html: str):
    os.makedirs(os.path.dirname(path_html), exist_ok=True)
    html_fragment = shap.plots.text(expl, display=False)
    if hasattr(html_fragment, "data"):
        html_fragment = html_fragment.data
    full_html = f"<!doctype html><html><head><meta charset='utf-8'>{shap.getjs()}</head>" \
                f"<body style='margin:0;padding:16px;font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif'>{html_fragment}</body></html>"
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(full_html)

def save_placeholder_html(path_html: str, message: str):
    os.makedirs(os.path.dirname(path_html), exist_ok=True)
    html_page = f"<!doctype html><html><head><meta charset='utf-8'><style>body{{font-family:system-ui; padding:16px}}</style></head><body><h3>{htmlmod.escape(message)}</h3></body></html>"
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(html_page)

# -------------------- index interattivo con TOGGLE --------------------
def build_clickable_index_html(
    tokenizer,
    y_ids: List[int],
    in_paths: List[str],
    pref_paths: List[str],
    out_html_path: str,
    input_prompt: str,
    title: str = "Token-level SHAP (Input / Prefisso)",
):
    spans = []
    visible_idxs = []
    for i, tid in enumerate(y_ids):
        s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if s == "":
            continue
        spans.append(f'<span class="tok" data-i="{i}">{htmlmod.escape(s)}</span>')
        visible_idxs.append(i)

    init_idx = visible_idxs[0] if visible_idxs else 0
    files_in = json.dumps(in_paths)
    files_pref = json.dumps(pref_paths)

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{htmlmod.escape(title)}</title>
<style>
body {{
  font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  margin: 0; padding: 16px;
}}
h1 {{ margin: 0 0 10px 0; }}
h2 {{ margin: 18px 0 10px 0; }}
.box {{
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  border: 1px solid #e5e7eb; border-radius: 10px;
  padding: 14px; background: #fafafa; margin-bottom: 14px;
  line-height: 1.45; max-height: 320px; overflow: auto;
}}
.outbox .tok {{ display:inline; cursor:pointer; border-radius:6px; padding:0 2px; }}
.outbox .tok:hover {{ background:#e6f0ff; }}
.outbox .tok.active {{ background:#cfe3ff; box-shadow:0 0 0 1px #99b3ff inset; }}
.controls {{ display:flex; gap:8px; align-items:center; margin: 6px 0 10px; }}
button.toggle {{ border:1px solid #d1d5db; background:#fff; padding:6px 10px; border-radius:8px; cursor:pointer; }}
button.toggle.active {{ background:#e6f0ff; border-color:#99b3ff; }}
.info {{ color:#6b7280; margin: 6px 0 12px 0; }}
iframe.plot {{ width:100%; height:640px; border:1px solid #e5e7eb; border-radius:10px; background:#fff; }}
</style>
<script>
let SOURCE = "input"; // "input" | "prefix"
const FILES_IN = {files_in};
const FILES_PREF = {files_pref};

function updateFrame(i) {{
  const frame = document.getElementById('plotframe');
  const info = document.getElementById('info');
  const files = (SOURCE === "input") ? FILES_IN : FILES_PREF;
  frame.src = files[i] || "";
  info.textContent = "Token selezionato: #" + i + " — sorgente: " + (SOURCE === "input" ? "Input" : "Prefisso output");
}}

function selectToken(i) {{
  document.querySelectorAll('.tok').forEach(t => t.classList.remove('active'));
  const el = document.querySelector('.tok[data-i="'+i+'"]');
  if (el) el.classList.add('active');
  updateFrame(i);
}}

function setSource(src) {{
  SOURCE = src;
  document.getElementById('btnInput').classList.toggle('active', src==="input");
  document.getElementById('btnPrefix').classList.toggle('active', src==="prefix");
  const cur = document.querySelector('.tok.active');
  const i = cur ? parseInt(cur.dataset.i, 10) : {init_idx};
  updateFrame(i);
}}

window.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('.tok').forEach(el => {{
    el.addEventListener('click', () => selectToken(parseInt(el.dataset.i, 10)));
  }});
  document.getElementById('btnInput').addEventListener('click', () => setSource("input"));
  document.getElementById('btnPrefix').addEventListener('click', () => setSource("prefix"));
  document.querySelector('.tok[data-i="{init_idx}"]')?.classList.add('active');
  setSource("input");
}});
</script>
</head>
<body>
  <h1>{htmlmod.escape(title)}</h1>

  <h2>Prompt di input</h2>
  <div class="box" id="inbox">{htmlmod.escape(input_prompt)}</div>

  <h2>Output generato (clicca un token)</h2>
  <div class="box outbox" id="outbox">{''.join(spans)}</div>

  <div class="controls">
    <span>Mostra contributi da:</span>
    <button id="btnInput" class="toggle active">Input</button>
    <button id="btnPrefix" class="toggle">Prefisso output</button>
  </div>

  <div class="info" id="info"></div>
  <iframe id="plotframe" class="plot" src=""></iframe>
</body>
</html>"""
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(page)

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser("SHAP token-level con toggle Input/Prefisso")
    parser.add_argument("--family", choices=["codegpt","codet5p"], required=True)
    parser.add_argument("--technique", choices=["full_ft","lora","prefix_tuning","prompt_tuning","adapters"], required=True)
    parser.add_argument("--checkpoint_path", type=str, required=False,
        help=("Per full_ft: path del modello finetunato (cartella con config+pesi). "
              "Per LoRA/prefix/prompt: path dell'adapter PEFT. "
              "Per adapters: path dell'adapter di AdapterHub."))
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--idx_sample", type=int, default=None)
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--max_tokens_to_explain", type=int, default=None)
    parser.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    parser.add_argument("--dtype", choices=["float32","float16","bfloat16"], default="float32")
    parser.add_argument("--out_dir", type=str, default="explain_token_shap_dual/out")
    parser.add_argument("--peft_merge", action="store_true",
        help="(Opzionale) PEFT merge_and_unload per fondere gli adapter nei pesi.")
    args = parser.parse_args()


    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # modello
    base_id = args.base_model or MODEL_HUB_ID[args.family]
    model, tokenizer = load_model_any_technique(
        family=args.family,
        technique=args.technique,
        checkpoint_path=args.checkpoint_path,
        base_model_hub_id=base_id,
        torch_dtype=torch_dtype,
        device=device,
        peft_merge=args.peft_merge,
    )

    # dataset
    raw = load_dataset("json", data_files={"test": args.test_file})["test"]
    texts = []
    for r in raw:
        t = record_to_last_user_text(r)
        if t:
            texts.append(t)
    if not texts:
        raise RuntimeError("Nessun testo (ultimo messaggio utente) trovato nel dataset.")
    if args.num_samples == 1 and args.idx_sample is not None:
        texts = [texts[args.idx_sample]]
        print(texts)
    else:
        texts = texts[:args.num_samples]
    logger.info(f"Esempi: {len(texts)}")

    masker_input = shap.maskers.Text(tokenizer)   
    masker_prefix = shap.maskers.Text(tokenizer)  

    for ex_idx, text in enumerate(texts):
        
        y_ids, y_str = generate_output_tokens(model, tokenizer, device, args.family, text, args.gen_max_new_tokens)
        
        
        if args.max_tokens_to_explain:
            y_ids = y_ids[:args.max_tokens_to_explain]
        if len(y_ids) == 0:
            logger.warning(f"[esempio {ex_idx}] nessun token generato.")
            continue
        
        if args.idx_sample is not None:
            base_dir = os.path.join(args.out_dir, f"{args.family}_{args.technique}", f"ex{args.idx_sample:03d}")    
        else:
            base_dir = os.path.join(args.out_dir, f"{args.family}_{args.technique}", f"ex{ex_idx:03d}")
        plots_in_dir = os.path.join(base_dir, "plots_input")
        plots_pref_dir = os.path.join(base_dir, "plots_prefix")
        os.makedirs(plots_in_dir, exist_ok=True)
        os.makedirs(plots_pref_dir, exist_ok=True)

        y_tokens = [tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False) for tid in y_ids]
        with open(os.path.join(base_dir, "y_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(y_tokens, f, ensure_ascii=False, indent=2)

        # salva input e output 
        with open(os.path.join(base_dir, "input.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(base_dir, "output_generated.txt"), "w", encoding="utf-8") as f:
            f.write(y_str)

        input_paths_rel, prefix_paths_rel = [], []

        for t in range(len(y_ids)):
            logger.info(f"[esempio {ex_idx}] Token {t}/{len(y_ids)-1}: calcolo SHAP (Input & Prefisso)")

            # --- Sorgente: Input ---
            predict_in = make_predict_fn_input_source(model, tokenizer, device, args.family, text, y_ids, t_index=t)
            explainer_in = shap.Explainer(predict_in, masker_input)
            sv_in = explainer_in([text], fixed_context=1)
            p_in = os.path.join(plots_in_dir, f"token_{t:03d}.html")
            save_text_plot_full(sv_in[0], p_in)
            input_paths_rel.append(os.path.relpath(p_in, start=base_dir).replace(os.sep, "/"))
           
            # --- Sorgente: Prefisso output ---
            p_pref = os.path.join(plots_pref_dir, f"token_{t:03d}.html")
            sv_pref_obj = None  # <--- NEW: terrà l'Explanation se esiste

            if t == 0:
                # Non esiste prefisso y_<t>
                save_placeholder_html(
                    p_pref,
                    "Nessun prefisso: questo è il primo token generato (y_<t> vuoto)."
                )
            else:
            
                predict_pref, prefix_obs_text = make_predict_fn_prefix_source(
                    model, tokenizer, device, args.family,
                    fixed_input_text=text, y_ids=y_ids, t_index=t
                )

                if (predict_pref is None) or _is_effectively_empty_text(prefix_obs_text):
                    # Prefisso vuoto o solo token speciali: nessuna feature da spiegare
                    save_placeholder_html(
                        p_pref,
                        "Prefisso vuoto o composto solo da token speciali: nessuna feature da spiegare."
                    )
                else:
                    explainer_pref = shap.Explainer(predict_pref, masker_prefix)
                    sv_pref = explainer_pref([prefix_obs_text], fixed_context=0)
                    save_text_plot_full(sv_pref[0], p_pref)
                    sv_pref_obj = sv_pref[0] 

            
            prefix_paths_rel.append(os.path.relpath(p_pref, start=base_dir).replace(os.sep, "/"))
            
           
            plots_in_dir   = os.path.join(base_dir, "plots_input")
            plots_pref_dir = os.path.join(base_dir, "plots_prefix")
            os.makedirs(plots_in_dir, exist_ok=True)
            os.makedirs(plots_pref_dir, exist_ok=True)

            # sottocartelle per i pickle
            pkls_in_dir    = os.path.join(base_dir, "pkls_input")
            pkls_pref_dir  = os.path.join(base_dir, "pkls_prefix")
            os.makedirs(pkls_in_dir, exist_ok=True)
            os.makedirs(pkls_pref_dir, exist_ok=True)


            # salva i pickle per debug in sottocartelle dedicate
            pkl_in_path = os.path.join(pkls_in_dir,   f"token_{t:03d}.pkl")
            with open(pkl_in_path, "wb") as f:
                pickle.dump(sv_in[0], f, protocol=pickle.HIGHEST_PROTOCOL)

            if sv_pref_obj is not None:
                pkl_pref_path = os.path.join(pkls_pref_dir, f"token_{t:03d}.pkl")
                with open(pkl_pref_path, "wb") as f:
                    pickle.dump(sv_pref_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            


        index_path = os.path.join(base_dir, "index.html")
        build_clickable_index_html(
            tokenizer=tokenizer,
            y_ids=y_ids,
            in_paths=input_paths_rel,
            pref_paths=prefix_paths_rel,
            out_html_path=index_path,
            input_prompt=text,
            title="Token-level SHAP — Input vs Prefisso output"
        )
        logger.info(f"[esempio {ex_idx}] pronto! Apri: {index_path}")

if __name__ == "__main__":
    main()

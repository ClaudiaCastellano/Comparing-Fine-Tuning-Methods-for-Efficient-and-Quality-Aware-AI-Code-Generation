import os, glob, json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# ============ Config pagina ============
st.set_page_config(page_title="Token-level SHAP Viewer", layout="wide")

# ============ Radice risultati  ============
RESULTS_ROOT = (Path(__file__).resolve().parent / "explain_token_shap_dual/out").resolve()

# ============ Utility ============
def discover_runs(root: Path):
    runs = {}
    if not root.exists(): return runs
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        ex_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("ex")])
        if ex_dirs: runs[run_dir.name] = ex_dirs
    return runs

def read_text(path: Path):
    try: return path.read_text(encoding="utf-8")
    except Exception: return ""

def count_token_plots(ex_dir: Path):
    return len(sorted(glob.glob(str(ex_dir / "plots_input" / "token_*.html"))))

def get_plot_html(ex_dir: Path, source: str, idx: int):
    sub = "plots_input" if source == "input" else "plots_prefix"
    path = ex_dir / sub / f"token_{idx:03d}.html"
    if not path.exists():
        return f"<html><body style='font-family:sans-serif;padding:1rem'>Plot non trovato: {path.name}</body></html>"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"<html><body style='font-family:sans-serif;padding:1rem'>Errore nel leggere {path.name}: {e}</body></html>"

def try_load_y_tokens(ex_dir: Path):
    p = ex_dir / "y_tokens.json"
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                toks = json.load(f)
            return [t if isinstance(t, str) else str(t) for t in toks]
        except Exception:
            pass
    return None

def align_labels_to_plots(labels, num_plots: int):
    if labels is None: labels = []
    if len(labels) >= num_plots:
        return labels[:num_plots]
    return labels + [""] * (num_plots - len(labels))  

def nice_label(s: str) -> str:
    if s is None or s == "" or s.strip() == "":
        return "⟨spazio⟩"
    return s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "⏎")

# ============ Sidebar ============
st.sidebar.title("Impostazioni")


runs = discover_runs(RESULTS_ROOT)
if not runs:
    st.error(f"Nessun esperimento trovato in: {RESULTS_ROOT}\n"
             f"Copia le cartelle dei risultati dentro `results/`.")
    st.stop()

run_names = sorted(runs.keys())
run_sel = st.sidebar.selectbox("Seleziona esperimento (run)", run_names)

ex_dirs = runs[run_sel]
ex_labels = [p.name for p in ex_dirs]
ex_sel_label = st.sidebar.selectbox("Seleziona esempio", ex_labels)
ex_sel = next(p for p in ex_dirs if p.name == ex_sel_label)

num_tokens = count_token_plots(ex_sel)
if num_tokens == 0:
    st.warning(f"Nessun plot token trovato in {ex_sel}/plots_input")
    st.stop()

# Etichette token 
labels = try_load_y_tokens(ex_sel)
labels = align_labels_to_plots(labels, num_tokens)


# Selezione da menu a tendina, mostrando parola + indice
opt_indices = list(range(num_tokens))
token_idx = st.sidebar.selectbox(
    "Token da spiegare (scegli la parola)",
    options=opt_indices,
    index=0,
    format_func=lambda i: f"{nice_label(labels[i])}  (#{i})"
)


view_mode = st.sidebar.radio("Vista", ["Input", "Prefisso", "Affianca"], horizontal=False)

# ============ Header ============
sel_word = nice_label(labels[token_idx])
st.title("Token-level SHAP Viewer")
st.caption(f"Run: `{run_sel}`  •  Esempio: `{ex_sel.name}`  •  Token selezionato: “{sel_word}”")

# ============ Prompt + Output ============
col1, col2 = st.columns(2)
with col1:
    st.subheader("Prompt di input")
    inbox = read_text(ex_sel / "input.txt")
    st.code(inbox or "(vuoto)", language=None)

with col2:
    st.subheader("Output generato")
    outbox = read_text(ex_sel / "output_generated.txt")
    st.code(outbox or "(vuoto)", language=None)

st.markdown("---")

# ============ Plot ============
def show_plot(source: str, idx: int, height: int = 720):
    html_plot = get_plot_html(ex_sel, source, idx)
    components.html(html_plot, height=height, scrolling=True)

if view_mode == "Affianca":
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Contributi da INPUT")
        show_plot("input", token_idx, height=720)
    with c2:
        st.subheader("Contributi da PREFISSO")
        show_plot("prefix", token_idx, height=720)
else:
    st.subheader(f"Contributi da {'INPUT' if view_mode=='Input' else 'PREFISSO'}")
    show_plot("input" if view_mode == "Input" else "prefix", token_idx, height=820)





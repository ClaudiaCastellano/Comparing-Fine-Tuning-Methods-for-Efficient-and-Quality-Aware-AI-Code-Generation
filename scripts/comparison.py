# File per confrontare i risultati delle varie tecniche
# Crea tabelle riassuntive in CSV e TXT

import os
import re
import json
import pandas as pd
from tabulate import tabulate  # per output txt formattato

# Tecniche e abbreviazioni
technique_map = {
    "pretrained": "pretrained",
    "full-finetuning": "full_ft",
    "lora": "lora",
    "adapters": "adapters",
    "prefix-tuning": "prefix_tuning",
    "prompt-tuning": "prompt_tuning"
}

contexts = ["secure", "insecure"]
#base_dir = "../codegpt/report" # per codegpt
base_dir = "../codeT5+/report"

if "codegpt" in base_dir.lower():
    results_dir = "results/comparison_tables_codegpt"
else:
    results_dir = "results/comparison_tables_codet5+"

os.makedirs(results_dir, exist_ok=True)


def parse_metrics(text):
    def extract(label, percent=True, decimals=2):
        match = re.search(rf"{label}:\s*([\d.]+)%", text)
        if match:
            value = float(match.group(1))
            return f"{round(value, decimals)}%" if percent else f"{round(value, decimals)}"
        return "N/A"

    return {
        "CrystalBLEU": extract("CrystalBLEU"),
        "BLEU-1": extract("BLEU-1"),
        "BLEU-2": extract("BLEU-2"),
        "BLEU-3": extract("BLEU-3"),
        "BLEU-4": extract("BLEU-4"),
        "ROUGE-L": extract("ROUGE-L"),
        "METEOR": extract("METEOR"),
        "EM": extract("EM"),
        "ED": extract("ED")
    }


def parse_semgrep(text):
    data = {}

    # Estrazione numerica base
    total_funcs = int(re.search(r"Total scanned functions:\s*(\d+)", text).group(1))
    clean_funcs = int(re.search(r"Total clean functions:\s*(\d+)", text).group(1))
    defective_funcs = int(re.search(r"Total defective functions.*?:\s*(\d+)", text).group(1))
    errors = int(re.search(r"Total errors:\s*(\d+)", text).group(1))
    empty_files = int(re.search(r"Total empty files:\s*(\d+)", text).group(1))
    total_issues = int(re.search(r"Total issues.*?:\s*(\d+)", text).group(1))
    errors_plus_empty = errors + empty_files

    # Estrazione issue per categoria
    cat_match = re.search(r"No of issues per category type:.*?{(.*?)}", text, re.DOTALL)
    issue_counts = eval("{" + cat_match.group(1) + "}") if cat_match else {}

    total_categorized = sum(issue_counts.values()) if issue_counts else 0
    def fmt(value):
        if total_categorized == 0:
            return f"{value} (0.0%)"
        return f"{value} ({round(100 * value / total_categorized, 2)}%)"

    # Costruzione delle colonne
    data["Total Functions"] = total_funcs
    data["Clean Functions"] = f"{clean_funcs} ({round(100 * clean_funcs / total_funcs, 2)}%)"
    data["Defective Functions"] = f"{defective_funcs} ({round(100 * defective_funcs / total_funcs, 2)}%)"
    data["Error Functions"] = errors
    data["Empty Files"] = empty_files
    data["Errors + Empty Files"] = f"{errors_plus_empty} ({round(100 * errors_plus_empty / total_funcs, 2)}%)"
    data["Total Issues"] = total_issues
    data["Security Issues"] = fmt(issue_counts.get("security", 0))
    data["Correctness Issues"] = fmt(issue_counts.get("correctness", 0))
    data["Maintainability Issues"] = fmt(issue_counts.get("maintainability", 0))
    data["Best-practice Issues"] = fmt(issue_counts.get("best-practice", 0))

    return data


def parse_resources(json_text):
    j = json.loads(json_text)
    return {
        "training_time_sec": j.get("training_time_sec"),
        "gpu_max_memory_GB": j.get("gpu_max_memory_GB"),
        "trainable_parameters": j.get("trainable_parameters"),
        "total_parameters": j.get("total_parameters"),
        "model_disk_size_MB": j.get("model_disk_size_MB")
    }

# Tabelle da popolare
metrics_rows = []
semgrep_rows = []
resources_rows = []

# --- METRICHE & SEMGREP ---
for tech_readable, tech_file in technique_map.items():
    for ctx in contexts:
        config_name = f"{tech_readable}-{ctx}"
        folder_name = f"{tech_file}_{ctx}"
        folder_path = os.path.join(base_dir, folder_name)

        # METRICHE
        metrics_file = os.path.join(folder_path, f"metriche_{tech_file}_{ctx}.txt")
        row_metrics = {"config": config_name}

        try:
            with open(metrics_file, "r") as f:
                metrics_text = f.read()
                row_metrics.update(parse_metrics(metrics_text))
        except FileNotFoundError:
            print(f"[!] Manca file metriche per {config_name}")
            continue

        metrics_rows.append(row_metrics)

        # SEMGREP
        row_semgrep = {"config": config_name}
        #semgrep_file = os.path.join(folder_path, f"{tech_file}_semgrep_{ctx}_report.txt") #codegpt
        semgrep_file = os.path.join(folder_path, f"{tech_file}_codet5+_semgrep_{ctx}_report.txt")
        try:
            with open(semgrep_file, "r") as f:
                semgrep_text = f.read()
                row_semgrep.update(parse_semgrep(semgrep_text))
        except FileNotFoundError:
            print(f"[!] Manca file semgrep per {config_name}")
        semgrep_rows.append(row_semgrep)

# --- RISORSE (una per tecnica) ---
for tech_readable, tech_file in technique_map.items():
    if tech_file == "pretrained":
        continue  # pretrained non ha risorse

    folder_path = os.path.join(base_dir, f"{tech_file}_secure")  # arbitrario
    resource_file = os.path.join(folder_path, "resource_report.json")
    row_resource = {"technique": tech_readable}

    try:
        with open(resource_file, "r") as f:
            json_text = f.read()
            row_resource.update(parse_resources(json_text))
    except FileNotFoundError:
        print(f"[!] Manca file risorse per {tech_readable}")

    resources_rows.append(row_resource)

# --- CREA DATAFRAME ---
df_metrics = pd.DataFrame(metrics_rows)
ordered_cols = ["config", "CrystalBLEU", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "EM", "ED"]
df_metrics = df_metrics[ordered_cols]
df_semgrep = pd.DataFrame(semgrep_rows)
df_resources = pd.DataFrame(resources_rows)

# --- SALVATAGGIO ---
df_metrics.to_csv(os.path.join(results_dir, "tabella_metriche.csv"), index=False)
with open(os.path.join(results_dir, "tabella_metriche.txt"), "w") as f:
    f.write(tabulate(df_metrics, headers="keys", tablefmt="grid", showindex=False))

df_semgrep.to_csv(os.path.join(results_dir, "tabella_semgrep.csv"), index=False)
with open(os.path.join(results_dir, "tabella_semgrep.txt"), "w") as f:
    f.write(tabulate(df_semgrep, headers="keys", tablefmt="grid", showindex=False))

df_resources.to_csv(os.path.join(results_dir, "tabella_risorse.csv"), index=False)
with open(os.path.join(results_dir, "tabella_risorse.txt"), "w") as f:
    f.write(tabulate(df_resources, headers="keys", tablefmt="grid", showindex=False))


# --- STAMPA SU TERMINALE ---
print("\n=== METRICHE ===")
print(tabulate(df_metrics, headers="keys", tablefmt="fancy_grid", showindex=False))

print("\n=== SEMGREP ===")
print(tabulate(df_semgrep, headers="keys", tablefmt="fancy_grid", showindex=False))

print("\n=== RISORSE ===")
print(tabulate(df_resources, headers="keys", tablefmt="fancy_grid", showindex=False))

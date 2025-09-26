# Comparing Fine-Tuning Methods for Efficient and Quality-Aware AI Code Generation

This repository contains the code and experiments developed for my Master's Thesis, focused on exploring and comparing **parameter-efficient fine-tuning (PEFT)** methods applied to **Code Large Language Models (Code LLMs)**.  
The goal of this work is to evaluate how different fine-tuning techniques impact **efficiency**, **code quality** and **security** in AI-based code generation.

---

## 📖 Overview

Large Language Models (LLMs) have rapidly transformed the landscape of software development by enabling automatic code generation, completion, and correction. However, these models are often computationally expensive to fine-tune and can introduce issues related to the **quality** and **security** of generated code.  

This thesis investigates whether **PEFT methods**, such as **Adapters**, **Prefix Tuning**, **Prompt Tuning** and **LoRA**, can provide effective alternatives to full fine-tuning, reducing resource consumption while maintaining or improving performance in terms of:

- **Generation quality**  
- **Code security**  
- **Training efficiency**

Two representative models were considered: **CodeT5+** and **CodeGPT**.

---

## 📂 Repository Structure

```
Comparing-Fine-Tuning-Methods-for-Efficient-and-Quality-Aware-AI-Code-Generation/
│
├── codeT5+/  # Experiments with CodeT5+ (LoRA, Prefix, Prompt, Full-FT, Pretrained)
│ 
├── codegpt/ # Experiments with CodeGPT (same structure)
│
├── datasets/ # Datasets (train, validation, test splits)
│
├── scripts/ # Utility scripts for evaluation and reporting
│ ├── output_similarity_metrics_best.py # Code quality metrics
│ ├── semgrep/ # Security analysis (Semgrep)
│ └── ... # Other helper scripts
│
├── requirements/ 
│
├── results/ 
│ ├── comparison_tables_codegpt/ # Summary tables of final results for quality, security and resource consumption
│ └── comparison_tables_codet5+
│
├── shap/ 
│ └── explain_token_shap_dual/out # SHAP explanation outputs
│ └── app_new.py # Streamlit app for interactive SHAP plots
│ └── full_suite_new.py # Script for generating explanation results
│
└── README.md # Project documentation
```

Each technique folder (`lora/`, `prefix_tuning/`, `prompt_tuning/`, `full_ft/`, `pretrained/`) contains:
- Training and inference scripts  
- Predictions (secure / insecure)  
- Training logs  

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ClaudiaCastellano/Comparing-Fine-Tuning-Methods-for-Efficient-and-Quality-Aware-AI-Code-Generation.git
cd Comparing-Fine-Tuning-Methods-for-Efficient-and-Quality-Aware-AI-Code-Generation
```

### 2. Create environment
It is recommended to use Python 3.9+ and a virtual environment.

By default, install dependencies from `requirements.txt`:

```bash
# Create a new conda environment
conda create -n thesis_env python=3.9

# Activate the environment
conda activate thesis_env

# Install required packages
pip install -r requirements.txt
```

⚠️ **Note on Adapters**

If you want to run experiments with Adapters, you will need an additional dependency:
```python
from transformers.adapters import AdapterConfig
```
Since this package created conflicts with the main environment, I used a separate conda environment dedicated to Adapters.
You can reproduce it by creating another environment and installing from `adapters_requirements.txt`:
```bash
conda create -n adapters_env python=3.9
conda activate adapters_env
pip install -r adapters_requirements.txt
```

### 3. Run experiments
Each technique has its own training and inference scripts inside `codeT5+` and `codegpt`.

⚠️ **Important**:  
- In `codeT5+`, scripts accept parameters via CLI flags (`--train_file`, `--eval_file`, etc.).  
- In `codegpt`, parameters must be **manually edited inside the training scripts** before running.  

Example with codeT5+ and LoRA:
```bash
# training
python codeT5+/lora/lora_codet5+.py --train_file datasets/train_datasets/train_secure.jsonl --eval_file datasets/validation_datasets/validation_secure.jsonl 

# inference
python codeT5+/lora/lora_inference_codet5+.py --input_file datasets/test_datasets/test_secure.jsonl --output_file predictions_secure.jsonl --model_name model_checkpoint/
```
Example with CodeGPT and Adapters: 
```bash
# training
python codegpt/adapters/adapters.py 

#inference
python codegpt/adapters/adapters_inference.py --input_file datasets/test_datasets/test_secure.jsonl --output_file predictions_secure.jsonl --model_name model_checkpoint/
```


## 📊 Evaluation

Evaluation is performed through the helper scripts in `scripts/`:

- **Quality**

  Edit the path of the predictions to analyze in the file `output_similarity_metrics_best.py`, then run:
  ```bash
  python output_similarity_metrics_best.py > metrics.txt
  ```
    

- **Security**
  - Extract generated code with jupyter notebook `scripts/semgrep/generated_code_extraction.ipynb`
  
  -  Run static analysis
      ```bash
      python scripts/semgrep/analyze_code_new.py extracted_code.json
      ```

  -  Process results
      ```bash 
      python scripts/process_results_new.py extracted_code_semgrep_result_batch <num_batches> \
      --empty_files <num_empty_files> > security_report.txt
        ```

- **Efficiency**

  Automatically logged by inference scripts (GPU/CPU time, memory, trainable parameters).

- **Comparison tables**
  ```bash
  python scripts/comparison.py
  ```
   Results are stored in `results/`.


---

## 🔍 SHAP Analysis

In addition to fine-tuning and evaluation, the repository includes a workflow for **explainability analysis** using **SHAP**:

- **`shap/full_suite_new.py`**  
  Runs the full SHAP analysis pipeline and generates explanation results.

- **`explain_token_shap_dual/out/`**  
  Stores SHAP results, including:
  - `.pkl` files with raw SHAP values  
  - `.html` reports with interactive explanations  

- **`shap/app_new.py`**  
  A **Streamlit application** to visualize SHAP explanations interactively.  
  Run it with:
  ```bash
  streamlit run shap/app_new.py
  ```


## 📑 Notes

- Datasets are split into train, validation and test sets.

- Reports with detailed metrics for each technique (secure/insecure) are stored in `reports/`.




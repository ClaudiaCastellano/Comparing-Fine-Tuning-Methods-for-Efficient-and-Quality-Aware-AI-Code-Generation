# File per filtrare i campioni migliori in base a EM e CrystalBLEU
# Mantiene i campioni con EM=1 o CrystalBLEU > 0.80
# Riscrive il file di output con i campioni selezionati, gli altri con campi vuoti
# Usa le funzioni di calcolo delle metriche da output_similarity_metrics.py

import sys
from crystal_bleu import compute_crystal_bleu, compute_trivially_shared_ngrams
import evaluate
import subprocess
from file_parser import read_json_singlefile
import pylcs
import os
import numpy as np
from rouge import Rouge

meteor = evaluate.load('meteor')
bleu = evaluate.load('bleu')
	
def edit_dist(hyp, ref):
	tmp = pylcs.edit_distance(hyp, ref)
	res_norm = 1-(tmp/max(len(hyp),len(ref)))
	return res_norm
 
def calc_ed(hyps, refs):
	scores = [edit_dist(h, r) for h, r in zip(hyps, refs)]
	mean_ed = np.mean(scores)
	min_ed = np.min(scores)
	max_ed = np.max(scores)
	median_ed = np.median(scores)
	q1_ed = np.percentile(scores, 25)
	q3_ed = np.percentile(scores, 75)
	formatted_score = (f'ED: {mean_ed * 100:.2f}% (min: {min_ed:.3f}, max: {max_ed:.3f}, median: {median_ed:.3f}, Q1: {q1_ed:.3f}, Q3: {q3_ed:.3f})')
	print(formatted_score)
	return formatted_score

def calc_meteor(hyps, refs):
	scores = [meteor.compute(predictions=[h], references=[r])['meteor'] for h, r in zip(hyps, refs)]
	mean_meteor = np.mean(scores)
	min_meteor = np.min(scores)
	max_meteor = np.max(scores)
	median_meteor = np.median(scores)
	q1_meteor = np.percentile(scores, 25)
	q3_meteor = np.percentile(scores, 75)
	formatted_score = (f'METEOR: {mean_meteor * 100:.2f}% (min: {min_meteor:.3f}, max: {max_meteor:.3f}, median: {median_meteor:.3f}, Q1: {q1_meteor:.3f}, Q3: {q3_meteor:.3f})')
	print(formatted_score)
	return formatted_score
	
def calc_rouge(hyps, refs):
	metrics = ["rouge-1","rouge-2","rouge-3","rouge-4","rouge-l"]
	all_f1_scores = {metric: [] for metric in metrics}
	formatted_score = []
	for i, metric in enumerate(metrics):
		rouge = Rouge(metrics=[metric])
		scores = rouge.get_scores(hyps, refs, avg=False)
		f1_scores = [score[metric]['f'] for score in scores]
		all_f1_scores[metric].extend(f1_scores)
		scores = np.array(all_f1_scores[metric])
		mean_rouge = np.mean(scores)
		min_rouge = np.min(scores)
		max_rouge = np.max(scores)
		median_rouge = np.median(scores)
		q1_rouge = np.percentile(scores, 25)
		q3_rouge = np.percentile(scores, 75)
		formatted_score.append(f'{metrics[i].upper()}: {mean_rouge * 100:.2f}% (min: {min_rouge:.3f}, max: {max_rouge:.3f}, median: {median_rouge:.3f}, Q1: {q1_rouge:.3f}, Q3: {q3_rouge:.3f})')
	for score in formatted_score:
		print(f"{score}" )
	return formatted_score
	
def calc_EM(hyps, refs):
	scores = [1 if hyp.split() == ref.split() else 0 for hyp, ref in zip(hyps, refs)]
	mean_em = np.mean(scores)
	formatted_score='EM: {0:.2f}%'.format(mean_em * 100)
	print(formatted_score)
	return formatted_score
        

def calc_corpus_BLEU(hyps, refs):
    from official.nlp.metrics import bleu
    hyps_tokenized = [bleu.bleu_tokenize(hyp) for hyp in hyps]
    refs_tokenized = [bleu.bleu_tokenize(ref) for ref in refs]

    formatted_score = []
    for i in range(1, 5):
        bleu_score = bleu.compute_bleu(refs_tokenized, hyps_tokenized, max_order=i, use_bp=True)
        formatted_score.append(f'BLEU-{i}:{bleu_score * 100:.2f}%')
    
    for score in formatted_score:
        print(score)
    return formatted_score

def calc_sentence_BLEU(hyps, refs):
	from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
	metrics = [(1,0,0,0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
	metric_name = ["BLEU-1: ", "BLEU-2: ", "BLEU-3: ", "BLEU-4: "]
	formatted_score = []
	for i, metric in enumerate(metrics):
		scores = []
		for hyp, ref in zip(hyps, refs):
			ref = ref.split()
			hyp = hyp.split()
			scores.append(sentence_bleu([ref], hyp, weights=metric, smoothing_function = SmoothingFunction().method1))
		mean_bleu = np.mean(scores)
		min_bleu = np.min(scores)
		max_bleu = np.max(scores)
		median_bleu = np.median(scores)
		q1_bleu = np.percentile(scores, 25)
		q3_bleu = np.percentile(scores, 75)
		formatted_score.append(f'{metric_name[i]}: {mean_bleu * 100:.2f}% (min: {min_bleu:.3f}, max: {max_bleu:.3f}, median: {median_bleu:.3f}, Q1: {q1_bleu:.3f}, Q3: {q3_bleu:.3f})')
	for score in formatted_score:
		print(f"{score}" )
	return formatted_score


def calc_crystalBLEU(hyps, refs, re_compute_ngrams: bool):
	cache_folder = "crystal_cache"
	if re_compute_ngrams:
		if not os.listdir(cache_folder):
			print("No files to delete. Will compute trivially shared ngrams")
		else:
			print("ngrams files deleted. Will compute trivially shared ngrams")
			files = os.listdir(cache_folder)
			for file in files:
				file_name = os.path.join(cache_folder, file)
				os.remove(file_name)
	else:
		print("Loading trivially shared ngrams")

	trivial_ngrams = compute_trivially_shared_ngrams(hyps, "python", cache_folder)
	scores = compute_crystal_bleu(refs, hyps, trivial_ngrams, "python")
	mean_crystal = np.mean(scores)
	min_crystal = np.min(scores)
	max_crystal = np.max(scores)
	median_crystal = np.median(scores)
	q1_crystal = np.percentile(scores, 25)
	q3_crystal = np.percentile(scores, 75)
	formatted_score = (f'\nCrystalBLEU: {mean_crystal * 100:.2f}% (min: {min_crystal:.3f}, max: {max_crystal:.3f}, median: {median_crystal:.3f}, Q1: {q1_crystal:.3f}, Q3: {q3_crystal:.3f})')
	print(formatted_score)
	return formatted_score
		

import json

if __name__ == '__main__':
	"""
		Read with the correct function to parse input file
	"""

	input_file = './adapters/adapters_predictions_secure.jsonl'
	hyps, refs = read_json_singlefile(input_file)
	print(f"Number of predictions: {len(hyps)}")
	print(f"Number of references: {len(refs)}")
	empty_hyps = [i for i, hyp in enumerate(hyps) if not hyp.strip()]
	print(f"Indici delle ipotesi vuote: {empty_hyps}")
	print(f"Number of empty predictions: {len(empty_hyps)}")

	for i in range(0, 10):
		print(f"Prediction: {hyps[i]}\n")
		print(f"Reference: {refs[i]}\n")
	
	# Calcolo metriche globali (solo per stampa)
	calc_crystalBLEU(hyps, refs, True)		
	calc_corpus_BLEU(hyps, refs)	
	calc_EM(hyps, refs)
	calc_ed(hyps, refs)
	#calc_rouge(hyps, refs)
	calc_meteor(hyps, refs)
	calc_sentence_BLEU(hyps, refs)

	# -------------------------
	# FILTRAGGIO SPECIFICO
	# -------------------------
		# -------------------------
	# FILTRAGGIO SPECIFICO
	# -------------------------
	print("\n🔎 Filtering samples with EM=1 or CrystalBLEU > 0.80")

	# Calcolo EM sample-wise
	em_scores = [1 if hyp.split() == ref.split() else 0 for hyp, ref in zip(hyps, refs)]

	# Calcolo CrystalBLEU sample-wise
	trivial_ngrams = compute_trivially_shared_ngrams(hyps, "python", "crystal_cache")
	crystal_scores = compute_crystal_bleu(refs, hyps, trivial_ngrams, "python")

	selected_indices = [i for i, (em, cb) in enumerate(zip(em_scores, crystal_scores)) if em == 1 or cb > 0.80]

	# Riscrivi tutte le righe, riempiendo i non selezionati con ""
	output_file = "selected_samples.jsonl"
	with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
		for i, line in enumerate(fin):
			if i in selected_indices:
				# Copia la riga originale
				fout.write(line)
			else:
				# Rimpiazza con campi vuoti
				try:
					obj = json.loads(line)
					if isinstance(obj, dict):
						for k in obj.keys():
							obj[k] = ""
						fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
					else:
						# fallback: stringa vuota
						fout.write("\"\"\n")
				except Exception:
					# se non è JSON valido, scrive stringa vuota
					fout.write("\"\"\n")

	print(f"✅ Saved {len(selected_indices)} selected samples (others blanked) to {output_file}")
	print("📌 Indici selezionati:", selected_indices)


	
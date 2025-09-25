from nltk.util import ngrams
from collections import Counter
from pygments.lexers import get_lexer_by_name
from crystalbleu import corpus_bleu
from datasets import load_dataset
import os, json, pickle

def compute_trivially_shared_ngrams(methods: list, language: str, cache_dir: str, ngrams_range: range = range(1, 5)) -> dict:
    """
    Calculate the trivially shared n-grams for the given methods.
    Load the trivially shared n-grams from file if already computed. Otherwise, compute them and save them on file.

    :param methods: The list of methods to analyze
    :param language: The language of the methods
    :param cache_dir: The directory to store the cache files
    :param ngrams_range: The range of n-grams to consider
    :return: The dictionary containing the trivially shared n-grams
    """

    ngrams_filename = f'{language}_trivially_shared_ngrams.pickle'
    ngrams_filepath = os.path.join(cache_dir, ngrams_filename)
    if os.path.exists(ngrams_filepath):
        print(f'Loading trivially shared n-grams from file {ngrams_filename} ...')
        with open(ngrams_filepath, 'rb') as f:
            return pickle.load(f)

    print(f'Computing trivially shared n-grams for language {language} ...')

    # Extract all n-grams of length 1-4
    k = 500
    all_ngrams = list()
    for idx, method in enumerate(methods):
        if not isinstance(method, str): continue
        if idx != 0 and idx % 100000 == 0:
            print(f'Processed {idx} methods')
        
        lexer = get_lexer_by_name(language)
        tokens = list(lexer.get_tokens(method))
        tokens = [token[1].strip() for token in tokens if token[1].strip()]
        for n in ngrams_range:
            all_ngrams.extend(list(ngrams(tokens, n)))

    # Calculate frequencies of all n-grams
    frequencies = Counter(all_ngrams)
    trivially_shared_ngrams = dict(frequencies.most_common(k))

    # Save trivially shared ngrams on file
    with open(ngrams_filepath, 'wb') as f:
        pickle.dump(trivially_shared_ngrams, f, protocol=pickle.HIGHEST_PROTOCOL)

    trivially_shared_ngrams_dict = dict()
    for k in trivially_shared_ngrams.keys():
        trivially_shared_ngrams_dict[str(k)]=k
        
    txt_filepath_ngrams = os.path.join(cache_dir, f'{language}_trivially_shared_ngrams.txt')
    with open(txt_filepath_ngrams, 'w') as convert_file:
        convert_file.write(json.dumps(trivially_shared_ngrams_dict))

    return trivially_shared_ngrams


def compute_crystal_bleu(references: list, candidates: list, trivial_ngrams: dict, language: str, weights: tuple = (1./4., 1./4., 1./4., 1./4.)) -> float:
    """
    Compute the CrystalBLEU score for the given references and candidates. 
    The references and candidates are tokenized using the lexer of the given language.

    :param references: The list of reference methods
    :param candidates: The list of candidate methods
    :param trivial_ngrams: The dictionary containing the trivially shared n-grams
    :param language: The language of the methods to analyze
    :param weights: The weights for the n-grams. Default is (1./4., 1./4., 1./4., 1./4.)
    :return: The list of CrystalBLEU scores for the given references and candidates
    """
    scores = list()
    lexer = get_lexer_by_name(language)
    for ref, cand, in zip(references, candidates):
        ref_tokens = ' '.join([t[1] for t in lexer.get_tokens(ref) if t[1].strip()]).split()
        cand_tokens = ' '.join([t[1] for t in lexer.get_tokens(cand) if t[1].strip()]).split()
        score = corpus_bleu([[ref_tokens]], [cand_tokens], weights=weights, ignoring=trivial_ngrams)
        scores.append(round(score, 2))
    return scores
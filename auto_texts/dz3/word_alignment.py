import sys
import numpy as np
from models import PriorModel
from models import TranslationModel
from utils import read_all_tokens, output_alignments_per_test_set

def collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model):
    "E-step: infer posterior distribution over each sentence pair and collect statistics."
    assert False, "Implement this."
    corpus_log_likelihood = 0.0
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        # 1. Infer posterior using models (see models.py).
        # 2. Collect statistics in models (see models.py).
        # 3. Update log prob.
        corpus_log_likelihood += log_likelihood
    return corpus_log_likelihood

def estimate_models(src_corpus, trg_corpus, prior_model, translation_model, num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        # E-step
        corpus_log_likelihood = collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model)
        # M-step
        prior_model.recompute_parameters()
        translation_model.recompute_parameters()
        if iteration > 0:
            print("corpus log likelihood: %1.3f" % corpus_log_likelihood)
    return prior_model, translation_model

def align_corpus(src_corpus, trg_corpus, prior_model, translation_model):
    "Align each sentence pair in the corpus in turn."
    alignments = []
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        posteriors, _ = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model)
        alignments.append(np.argmax(posteriors, 0))
    return alignments

def initialize_models(src_corpus, trg_corpus):
    prior_model = PriorModel(src_corpus, trg_corpus)
    translation_model = TranslationModel(src_corpus, trg_corpus)
    return prior_model, translation_model

def normalize(src_corpus, trg_corpus):
    return src_corpus, trg_corpus

if __name__ == "__main__":
    if not len(sys.argv) == 5:
        print("Usage ./align.py src_corpus trg_corpus iterations output_prefix.")
        sys.exit(0)
    src_corpus, trg_corpus = read_all_tokens(sys.argv[1]), read_all_tokens(sys.argv[2])
    src_corpus, trg_corpus = normalize(src_corpus, trg_corpus)
    num_iterations = int(sys.argv[3])
    output_prefix = sys.argv[4]
    assert len(src_corpus) == len(trg_corpus), "Corpora should be same size!"
    prior_model, translation_model = initialize_models(src_corpus, trg_corpus)
    prior_model, translation_model = estimate_models(src_corpus, trg_corpus, prior_model, translation_model, num_iterations)    
    alignments = align_corpus(src_corpus, trg_corpus, prior_model, translation_model)
    output_alignments_per_test_set(alignments, output_prefix)

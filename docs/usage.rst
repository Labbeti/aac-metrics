Usage
========================

Evaluate default AAC metrics
############################

The full evaluation process to compute AAC metrics can be done with `aac_metrics.aac_evaluate` function.

.. code-block:: python

    from aac_metrics import aac_evaluate

    candidates: list[str] = ["a man is speaking", ...]
    mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ...]

    corpus_scores, _ = aac_evaluate(candidates, mult_references)
    print(corpus_scores)
    # dict containing the score of each aac metric: "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider_d", "spice", "spider"
    # {"bleu_1": tensor(0.7), "bleu_2": ..., ...}


Evaluate a specific metric
##########################

Evaluate a specific metric can be done using the `aac_metrics.functional.<metric_name>.<metric_name>` function or the `aac_metrics.classes.<metric_name>.<metric_name>` class.

.. warning::
    Unlike `aac_evaluate`, the tokenization with PTBTokenizer is not done with these functions, but you can do it manually with `preprocess_mono_sents` and `preprocess_mult_sents` functions.

.. code-block:: python
    
    from aac_metrics.functional import cider_d
    from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents

    candidates: list[str] = ["a man is speaking", ...]
    mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ...]

    candidates = preprocess_mono_sents(candidates)
    mult_references = preprocess_mult_sents(mult_references)

    corpus_scores, sents_scores = cider_d(candidates, mult_references)
    print(corpus_scores)
    # {"cider_d": tensor(0.1)}
    print(sents_scores)
    # {"cider_d": tensor([0.9, ...])}


Each metrics also exists as a python class version, like `aac_metrics.classes.cider_d.CIDErD`.

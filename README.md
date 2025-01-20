<!-- # -*- coding: utf-8 -*- -->

<div align="center">

# Audio Captioning metrics (aac-metrics)

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10.1+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
<a href="https://github.com/Labbeti/aac-metrics/actions"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/Labbeti/aac-metrics/python-package-pip.yaml?branch=main&style=for-the-badge&logo=github"></a>
<a href='https://aac-metrics.readthedocs.io/en/stable/?badge=stable'>
    <img src='https://readthedocs.org/projects/aac-metrics/badge/?version=stable&style=for-the-badge' alt='Documentation Status' />
</a>

Metrics for evaluating Automated Audio Captioning systems, designed for PyTorch.

</div>

## Why using this package?
- **Easy to install and download**
- **Produces same results than [caption-evaluation-tools](https://github.com/audio-captioning/caption-evaluation-tools), [fense](https://github.com/blmoistawinde/fense) and [mace](https://github.com/satvik-dixit/mace) repositories**
- **Provides 15 different metrics:**
    - BLEU [[1]](#bleu)
    - ROUGE-L [[2]](#rouge-l)
    - METEOR [[3]](#meteor)
    - CIDEr-D [[4]](#cider)
    - SPICE [[5]](#spice)
    - SPIDEr [[6]](#spider)
    - BERTScore [[7]](#bertscore)
    - SPIDEr-max [[8]](#spider-max)
    - SBERT-sim [[9]](#fense)
    - FER [[9]](#fense)
    - FENSE [[9]](#fense)
    - SPIDEr-FL [[10]](#spider-fl)
    - CLAP-sim [[11]](#clap-sim) :new:
    - MACE [[12]](#mace) :new:
    - Vocab (unique word vocabulary absolute or relative to references)

## Installation
Install the pip package:
```bash
pip install aac-metrics
```

If you want to check if the package has been installed and the version, you can use this command:
```bash
aac-metrics-info
```

Download the external code and models needed for METEOR, SPICE, SPIDEr, SPIDEr-max, PTBTokenizer, SBERTSim, FER, FENSE and SPIDEr-FL:
```bash
aac-metrics-download
```

Notes:
- The external code for SPICE, METEOR and PTBTokenizer is stored in `~/.cache/aac-metrics`.
- The weights of the FENSE fluency error detector and the the SBERT model are respectively stored by default in `~/.cache/torch/hub/fense_data` and `~/.cache/torch/sentence_transformers`.

## Usage
### Evaluate default metrics
The full evaluation pipeline to compute AAC metrics can be done with `aac_metrics.evaluate` function.

```python
from aac_metrics import evaluate

candidates: list[str] = ["a man is speaking", "rain falls"]
mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ["rain is falling hard on a surface"]]

corpus_scores, _ = evaluate(candidates, mult_references)
print(corpus_scores)
# dict containing the score of each metric: "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider_d", "spice", "spider"
# {"bleu_1": tensor(0.4278), "bleu_2": ..., ...}
```
### Evaluate DCASE2024 metrics
To compute metrics for the DCASE2023 challenge, just set the argument `metrics="dcase2024"` in `evaluate` function call.

```python
corpus_scores, _ = evaluate(candidates, mult_references, metrics="dcase2024")
print(corpus_scores)
# dict containing the score of each metric: "meteor", "cider_d", "spice", "spider", "spider_fl", "fer", "fense", "vocab"
```

### Evaluate a specific metric
Evaluate a specific metric can be done using the `aac_metrics.functional.<metric_name>.<metric_name>` function or the `aac_metrics.classes.<metric_name>.<metric_name>` class. Unlike `evaluate`, the tokenization with PTBTokenizer is not done with these functions, but you can do it manually with `preprocess_mono_sents` and `preprocess_mult_sents` functions.

```python
from aac_metrics.functional import cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents

candidates: list[str] = ["a man is speaking", "rain falls"]
mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ["rain is falling hard on a surface"]]

candidates = preprocess_mono_sents(candidates)
mult_references = preprocess_mult_sents(mult_references)

corpus_scores, sents_scores = cider_d(candidates, mult_references)
print(corpus_scores)
# {"cider_d": tensor(0.9614)}
print(sents_scores)
# {"cider_d": tensor([1.3641, 0.5587])}
```

Each metrics also exists as a python class version, like `aac_metrics.classes.cider_d.CIDErD`.

## Which metric(s) should I choose for Automated Audio Captioning?
To evaluate audio captioning systems, I would recommand to compute `SPIDEr`, `FENSE` and `Vocab` metrics. `SPIDEr` is useful to compare with the rest of the litterature, but it is highly sensitive to n-gram matching and can overestimate model trained with reinforcement learning. `FENSE` is more consistent and variable than `SPIDEr`, but it uses a model not trained on audio captions. `Vocab` can give you an insight about the model diversity. To compute all of these metrics at once, you can use for example the `Evaluate` class:

```python
from aac_metrics import Evaluate

evaluate = Evaluate(metrics=["spider", "fense", "vocab"])

candidates: list[str] = ...
mult_references: list[list[str]] = ...

corpus_scores, _ = evaluate(candidates, mult_references)

vocab_size = corpus_scores["vocab.cands"]
spider_score = corpus_scores["spider"]
fense_score = corpus_scores["fense"]
```

## Metrics
### Legacy metrics
| Metric name | Python Class | Origin | Range | Short description |
|:---|:---|:---|:---|:---|
| BLEU [[1]](#bleu) | `BLEU` | machine translation | [0, 1] | Precision of n-grams |
| ROUGE-L [[2]](#rouge-l) | `ROUGEL` | text summarization | [0, 1] | FScore of the longest common subsequence |
| METEOR [[3]](#meteor) | `METEOR` | machine translation | [0, 1] | Cosine-similarity of frequencies with synonyms matching |
| CIDEr-D [[4]](#cider) | `CIDErD` | image captioning | [0, 10] | Cosine-similarity of TF-IDF computed on n-grams |
| SPICE [[5]](#spice) | `SPICE` | image captioning | [0, 1] | FScore of a semantic graph |
| SPIDEr [[6]](#spider) | `SPIDEr` | image captioning | [0, 5.5] | Mean of CIDEr-D and SPICE |
| BERTScore [[7]](#bertscore) | `BERTScoreMRefs` | text generation | [0, 1] | Fscore of BERT embeddings. In contrast to torchmetrics, it supports multiple references per file. |

### AAC-specific metrics
| Metric name | Python Class | Origin | Range | Short description |
|:---|:---|:---|:---|:---|
| SPIDEr-max [[8]](#spider-max) | `SPIDErMax` | audio captioning | [0, 5.5] | Max of SPIDEr scores for multiples candidates |
| SBERT-sim [[9]](#fense) | `SBERTSim` | audio captioning | [-1, 1] | Cosine-similarity of **Sentence-BERT embeddings** |
| Fluency Error Rate [[9]](#fense) | `FER` | audio captioning | [0, 1] | Detect fluency errors in sentences with a pretrained model |
| FENSE [[9]](#fense) | `FENSE` | audio captioning | [-1, 1] | Combines SBERT-sim and Fluency Error rate |
| SPIDEr-FL [[10]](#spider-fl) | `SPIDErFL` | audio captioning | [0, 5.5] | Combines SPIDEr and Fluency Error rate |
| CLAP-sim [[11]](#clap-sim) | `CLAPSim` | audio captioning | [-1, 1] | Combines SPIDEr and Fluency Error rate |
| MACE [[12]](#spider-fl) | `MACE` | audio captioning | [-1, 1] | Combines CLAP-sim and Fluency Error rate |

### Other metrics
| Metric name | Python Class | Origin | Range | Short description |
|:---|:---|:---|:---|:---|
| Vocabulary | `Vocab` | text generation | [0, +&infin;[ | Number of unique words in candidates. |

### Future directions
This package currently does not include all metrics dedicated to audio captioning. Feel free to do a pull request / or ask to me by email if you want to include them. Those metrics not included are listed here:
- CB-Score [[13]](#cb-score)
- SPICE+ [[14]](#spice-plus)
- ACES [[15]](#aces) (can be found here: https://github.com/GlJS/ACES)
- SBF [[16]](#sbf)
- s2v [[17]](#s2v)

## Requirements
This package has been developped for Ubuntu 20.04, and it is expected to work on most Linux distributions.
<!-- Windows is not officially supported. -->

### Python packages

The pip requirements are automatically installed when using `pip install` on this repository.
```
torch >= 1.10.1
numpy >= 1.21.2
pyyaml >= 6.0
tqdm >= 4.64.0
sentence-transformers >= 2.2.2
transformers
torchmetrics >= 0.11.4
```

### External requirements
- `java` **>= 1.8 and <= 1.13** is required to compute METEOR, SPICE and use the PTBTokenizer.
Most of these functions can specify a java executable path with `java_path` argument or by overriding `AAC_METRICS_JAVA_PATH` environment variable.
<!-- sudo apt install default-jre -->

## Additional notes
### CIDEr or CIDEr-D?
The CIDEr metric differs from CIDEr-D because it applies a stemmer to each word before computing the n-grams of the sentences. In AAC, only the CIDEr-D is reported and used for SPIDEr in [caption-evaluation-tools](https://github.com/audio-captioning/caption-evaluation-tools), despite some papers called it "CIDEr".

### Do metrics work on multi-GPU?
No. Most of these metrics use numpy or external java programs to run, which prevents multi-GPU testing in parallel.

### Do metrics work on Windows/Mac OS?
Maybe. Most of the metrics only need python to run, which can be done on Windows/Mac. However, you might expect errors with METEOR metric, SPICE-based metrics and PTB tokenizer, since they requires an external java program to run. Feel free to open an issue if a metric does not work properly.

## About SPIDEr-max metric
SPIDEr-max [[7]](#spider-max) is a metric based on SPIDEr that takes into account multiple candidates for the same audio. It computes the maximum of the SPIDEr scores for each candidate to balance the high sensitivity to the frequency of the words generated by the model. For more detail, please see the [documentation about SPIDEr-max](https://aac-metrics.readthedocs.io/en/stable/spider_max.html).

## References

Citation in bibtex format are available in the following file in this repository: "[data/papers.bib](https://github.com/Labbeti/aac-metrics/blob/main/data/papers.bib)".

#### BLEU
[1] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: a method for automatic evaluation of machine translation,” in Proceedings of the 40th Annual Meeting on Association for Computational Linguistics - ACL ’02. Philadelphia, Pennsylvania: Association for Computational Linguistics, 2001, p. 311. [Online]. Available: http://portal.acm.org/citation.cfm?doid=1073083.1073135.

#### ROUGE-L
[2] C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in Text Summarization Branches Out. Barcelona, Spain: Association for Computational Linguistics, Jul. 2004, pp. 74–81. [Online]. Available: https://aclanthology.org/W04-1013

#### METEOR
[3] M. Denkowski and A. Lavie, “Meteor Universal: Language Specific Translation Evaluation for Any Target Language,” in Proceedings of the Ninth Workshop on Statistical Machine Translation. Baltimore, Maryland, USA: Association for Computational Linguistics, 2014, pp. 376–380. [Online]. Available: http://aclweb.org/anthology/W14-3348

#### CIDEr
[4] R. Vedantam, C. L. Zitnick, and D. Parikh, “CIDEr: Consensus-based Image Description Evaluation,” arXiv:1411.5726 [cs], Jun. 2015, [Online]. Available: http://arxiv.org/abs/1411.5726

#### SPICE
[5] P. Anderson, B. Fernando, M. Johnson, and S. Gould, “SPICE: Semantic Propositional Image Caption Evaluation,” arXiv:1607.08822 [cs], Jul. 2016, [Online]. Available: http://arxiv.org/abs/1607.08822

#### SPIDEr
[6] S. Liu, Z. Zhu, N. Ye, S. Guadarrama, and K. Murphy, “Improved Image Captioning via Policy Gradient optimization of SPIDEr,” 2017 IEEE International Conference on Computer Vision (ICCV), pp. 873–881, Oct. 2017, arXiv: 1612.00370. [Online]. Available: https://arxiv.org/abs/1612.00370

#### BERTScore
[7] T. Zhang*, V. Kishore*, F. Wu*, K. Q. Weinberger, and Y. Artzi, “BERTScore: Evaluating Text Generation with BERT,” 2020. [Online]. Available: https://openreview.net/forum?id=SkeHuCVFDr

#### SPIDEr-max
[8] E. Labbé, T. Pellegrini, and J. Pinquier, “Is my automatic audio captioning system so bad? spider-max: a metric to consider several caption candidates,” Nov. 2022. [Online]. Available: https://hal.archives-ouvertes.fr/hal-03810396

#### FENSE
[9] Z. Zhou, Z. Zhang, X. Xu, Z. Xie, M. Wu, and K. Q. Zhu, Can Audio Captions Be Evaluated with Image Caption Metrics? arXiv, 2022. [Online]. Available: http://arxiv.org/abs/2110.04684

#### SPIDEr-FL
[10] DCASE2023 website task6a description (ranking metric): https://dcase.community/challenge2023/task-automated-audio-captioning#evaluation

#### CLAP-sim
[11] B. Elizalde, S. Deshmukh, and H. Wang, Natural Language Supervision for General-Purpose Audio Representations. 2023. [Online]. Available: https://arxiv.org/abs/2309.05767

#### MACE
[12] S. Dixit, S. Deshmukh, and B. Raj, MACE: Leveraging Audio for Evaluating Audio Captioning Systems. 2024. [Online]. Available: https://arxiv.org/abs/2411.00321

#### CB-score
[13] I. Martín-Morató, M. Harju, and A. Mesaros, “A Summarization Approach to Evaluating Audio Captioning,” Nov. 2022. [Online]. Available: https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Martin-Morato_35.pdf

#### SPICE-plus
[14] F. Gontier, R. Serizel, and C. Cerisara, “SPICE+: Evaluation of Automatic Audio Captioning Systems with Pre-Trained Language Models,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5. doi: 10.1109/ICASSP49357.2023.10097021.

#### ACES
[15] G. Wijngaard, E. Formisano, B. L. Giordano, M. Dumontier, “ACES: Evaluating Automated Audio Captioning Models on the Semantics of Sounds”, in EUSIPCO 2023, 2023. Available: https://ieeexplore.ieee.org/document/10289793.

#### SBF
[16] R. Mahfuz, Y. Guo, A. K. Sridhar, and E. Visser, Detecting False Alarms and Misses in Audio Captions. 2023. [Online]. Available: https://arxiv.org/pdf/2309.03326.pdf

#### s2v
[17] S. Bhosale, R. Chakraborty, and S. K. Kopparapu, “A Novel Metric For Evaluating Audio Caption Similarity,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5. doi: 10.1109/ICASSP49357.2023.10096526. Available: https://ieeexplore.ieee.org/document/10096526

## Citation
If you use **SPIDEr-max**, you can cite the following paper using BibTex :
```bibtex
@inproceedings{Labbe2022,
    title        = {Is my Automatic Audio Captioning System so Bad? SPIDEr-max: A Metric to Consider Several Caption Candidates},
    author       = {Labb\'{e}, Étienne and Pellegrini, Thomas and Pinquier, Julien},
    year         = 2022,
    month        = {November},
    booktitle    = {Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022)},
    address      = {Nancy, France},
    url          = {https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Labbe_46.pdf}
}
```

If you use this software, please consider cite it as "Labbe, E. (2025). aac-metrics: Metrics for evaluating Automated Audio Captioning systems for PyTorch.", or use the following BibTeX citation:

```bibtex
@software{Labbe_aac_metrics_2025,
    author = {Labbé, Étienne},
    license = {MIT},
    month = {01},
    title = {{aac-metrics}},
    url = {https://github.com/Labbeti/aac-metrics/},
    version = {0.5.5},
    year = {2025},
}
```

## Contact
Maintainer:
- [Étienne Labbé](https://labbeti.github.io/) "Labbeti": labbeti.pub@gmail.com

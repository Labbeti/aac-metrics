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
- **Easy installation and download**
- **Same results than [caption-evaluation-tools](https://github.com/audio-captioning/caption-evaluation-tools) and [fense](https://github.com/blmoistawinde/fense) repositories**
- **Provides the following metrics:**
    - BLEU [[1]](#bleu)
    - ROUGE-L [[2]](#rouge-l)
    - METEOR [[3]](#meteor)
    - CIDEr-D [[4]](#cider)
    - SPICE [[5]](#spice)
    - SPIDEr [[6]](#spider)
    - SPIDEr-max [[7]](#spider-max)
    - SBERT [[8]](#fense)
    - FluencyError [[8]](#fense)
    - FENSE [[8]](#fense)
    - SPIDErErr

## Installation
Install the pip package:
```bash
pip install aac-metrics
```

Download the external code and models needed for METEOR, SPICE, PTBTokenizer and FENSE:
```bash
aac-metrics-download
```

Notes:
- The external code for SPICE, METEOR and PTBTokenizer is stored in `$HOME/.cache/aac-metrics`.
- The weights of the FENSE fluency error detector and the the SBERT model are respectively stored by default in `$HOME/.cache/torch/hub/fense_data` and `$HOME/.cache/torch/sentence_transformers`.

## Usage
### Evaluate default AAC metrics
The full evaluation process to compute AAC metrics can be done with `aac_metrics.aac_evaluate` function.

```python
from aac_metrics import aac_evaluate

candidates: list[str] = ["a man is speaking"]
mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"]]

corpus_scores, _ = aac_evaluate(candidates, mult_references)
print(corpus_scores)
# dict containing the score of each aac metric: "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider_d", "spice", "spider"
# {"bleu_1": tensor(0.7), "bleu_2": ..., ...}
```

### Evaluate a specific metric
Evaluate a specific metric can be done using the `aac_metrics.functional.<metric_name>.<metric_name>` function or the `aac_metrics.classes.<metric_name>.<metric_name>` class. Unlike `aac_evaluate`, the tokenization with PTBTokenizer is not done with these functions, but you can do it manually with `preprocess_mono_sents` and `preprocess_mult_sents` functions.

```python
from aac_metrics.functional import cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents

candidates: list[str] = ["a man is speaking"]
mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"]]

candidates = preprocess_mono_sents(candidates)
mult_references = preprocess_mult_sents(mult_references)

corpus_scores, sents_scores = cider_d(candidates, mult_references)
print(corpus_scores)
# {"cider_d": tensor(0.1)}
print(sents_scores)
# {"cider_d": tensor([0.9, ...])}
```

Each metrics also exists as a python class version, like `aac_metrics.classes.cider_d.CIDErD`.

## Metrics
### Default AAC metrics
| Metric | Python Class | Origin | Range | Short description |
|:---|:---|:---|:---|:---|
| BLEU [[1]](#bleu) | `BLEU` | machine translation | [0, 1] | Precision of n-grams |
| ROUGE-L [[2]](#rouge-l) | `ROUGEL` | machine translation | [0, 1] | FScore of the longest common subsequence |
| METEOR [[3]](#meteor) | `METEOR` | machine translation | [0, 1] | Cosine-similarity of frequencies with synonyms matching |
| CIDEr-D [[4]](#cider) | `CIDErD` | image captioning | [0, 10] | Cosine-similarity of TF-IDF computed on n-grams |
| SPICE [[5]](#spice) | `SPICE` | image captioning | [0, 1] | FScore of semantic graph |
| SPIDEr [[6]](#spider) | `SPIDEr` | image captioning | [0, 5.5] | Mean of CIDEr-D and SPICE |

### Other metrics
| Metric name | Python Class | Origin | Range | Short description |
|:---|:---|:---|:---|:---|
| SPIDEr-max [[7]](#spider-max) | `SPIDErMax` | audio captioning | [0, 5.5] | Max of SPIDEr scores for multiples candidates |
| SBERT [[7]](#spider-max) | `SBERT` | audio captioning | [-1, 1] | Cosine-similarity of **Sentence-BERT embeddings** |
| FluencyError [[7]](#spider-max) | `FluencyError` | audio captioning | [0, 1] | Use pretrained model to detect fluency errors in sentences |
| FENSE [[8]](#fense) | `FENSE` | audio captioning | [-1, 1] | Combines `SBERT` and `FluencyError` |
| SPIDErErr | `SPIDErErr` | audio captioning | [0, 5.5] | Combines `SPIDEr` and `FluencyError` |

## SPIDEr-max metric
SPIDEr-max [[7]](#spider-max)  is a metric based on SPIDEr that takes into account multiple candidates for the same audio. It computes the maximum of the SPIDEr scores for each candidate to balance the high sensitivity to the frequency of the words generated by the model.

### SPIDEr-max: why ?
The SPIDEr metric used in audio captioning is highly sensitive to the frequencies of the words used.

Here is 2 examples with the 5 candidates generated by the beam search algorithm, their corresponding SPIDEr scores and the associated references:

<div align="center">

| Beam search candidates | SPIDEr |
|:---|:---:|
| heavy rain is falling on a roof | 0.562 |
| heavy rain is falling on **a tin roof** | **0.930** |
| a heavy rain is falling on a roof | 0.594 |
| a heavy rain is falling on the ground | 0.335 |
| a heavy rain is falling on the roof | 0.594 |

| References |
|:---|
| heavy rain falls loudly onto a structure with a thin roof |
| heavy rainfall falling onto a thin structure with a thin roof |
| it is raining hard and the rain hits **a tin roof** |
| rain that is pouring down very hard outside |
| the hard rain is noisy as it hits **a tin roof** |

_(Candidates and references for the Clotho development-testing file named "rain.wav")_

| Beam search candidates | SPIDEr |
|:---|:---:|
| a woman speaks and a sheep bleats | 0.190 |
| a woman **speaks and a goat bleats** | **1.259** |
| a man speaks and a sheep bleats | 0.344 |
| an adult male speaks and a sheep bleats | 0.231 |
| an adult male is speaking and a sheep bleats | 0.189 |

| References |
|:---|
| a man speaking and laughing followed by a goat bleat |
| a man is speaking in high tone while a goat is bleating one time |
| a man speaks followed by a goat bleat |
| a person **speaks and a goat bleats** |
| a man is talking and snickering followed by a goat bleating |

_(Candidates and references for an AudioCaps testing file with the id "jid4t-FzUn0")_
</div>

Even with very similar candidates, the SPIDEr scores varies drastically. To adress this issue, we proposed a SPIDEr-max metric which take the maximum value of several candidates for the same audio. SPIDEr-max demonstrate that SPIDEr can exceed state-of-the-art scores on AudioCaps and Clotho and even human scores on AudioCaps [[7]](#spider-max).

### SPIDEr-max: usage
This usage is very similar to other captioning metrics, with the main difference of take a multiple candidates list as input.

```python
from aac_metrics.functional import spider_max
from aac_metrics.utils.tokenization import preprocess_mult_sents

mult_candidates: list[list[str]] = [["a man is speaking", "maybe someone speaking"]]
mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"]]

mult_candidates = preprocess_mult_sents(mult_candidates)
mult_references = preprocess_mult_sents(mult_references)

corpus_scores, sents_scores = spider_max(mult_candidates, mult_references)
print(corpus_scores)
# {"spider": tensor(0.1), ...}
print(sents_scores)
# {"spider": tensor([0.9, ...]), ...}
```

## Requirements
### Python packages

The pip requirements are automatically installed when using `pip install` on this repository.
```
torch >= 1.10.1
numpy >= 1.21.2
pyyaml >= 6.0
tqdm >= 4.64.0
sentence-transformers>=2.2.2
```

### External requirements
- `java` >= 1.8 is required to compute METEOR, SPICE and use the PTBTokenizer.
Most of these functions can specify a java executable path with `java_path` argument.

- `unzip` command to extract SPICE zipped files.


## Additional notes
### CIDEr or CIDEr-D ?
The CIDEr metric differs from CIDEr-D because it applies a stemmer to each word before computing the n-grams of the sentences. In AAC, only the CIDEr-D is reported and used for SPIDEr in [caption-evaluation-tools](https://github.com/audio-captioning/caption-evaluation-tools), but some papers called it "CIDEr".

### Does metrics work on multi-GPU ?
No. Most of these metrics use numpy or external java programs to run, which prevents multi-GPU testing for now.

### Is torchmetrics needed for this package ?
No. But if torchmetrics is installed, all metrics classes will inherit from the base class `torchmetrics.Metric`.
It is because most of the metrics does not use PyTorch tensors to compute scores and numpy and strings cannot be added to states of `torchmetrics.Metric`.

## References
#### BLEU
[1] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: a
method for automatic evaluation of machine translation,” in Proceed-
ings of the 40th Annual Meeting on Association for Computational
Linguistics - ACL ’02. Philadelphia, Pennsylvania: Association
for Computational Linguistics, 2001, p. 311. [Online]. Available:
http://portal.acm.org/citation.cfm?doid=1073083.1073135

#### ROUGE-L
[2] C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,”
in Text Summarization Branches Out. Barcelona, Spain: Association
for Computational Linguistics, Jul. 2004, pp. 74–81. [Online]. Available:
https://aclanthology.org/W04-1013

#### METEOR
[3] M. Denkowski and A. Lavie, “Meteor Universal: Language Specific
Translation Evaluation for Any Target Language,” in Proceedings of the
Ninth Workshop on Statistical Machine Translation. Baltimore, Maryland,
USA: Association for Computational Linguistics, 2014, pp. 376–380.
[Online]. Available: http://aclweb.org/anthology/W14-3348

#### CIDEr
[4] R. Vedantam, C. L. Zitnick, and D. Parikh, “CIDEr: Consensus-based
Image Description Evaluation,” arXiv:1411.5726 [cs], Jun. 2015, arXiv:
1411.5726. [Online]. Available: http://arxiv.org/abs/1411.5726

#### SPICE
[5] P. Anderson, B. Fernando, M. Johnson, and S. Gould, “SPICE: Semantic
Propositional Image Caption Evaluation,” arXiv:1607.08822 [cs], Jul. 2016,
arXiv: 1607.08822. [Online]. Available: http://arxiv.org/abs/1607.08822

#### SPIDEr
[6] S. Liu, Z. Zhu, N. Ye, S. Guadarrama, and K. Murphy, “Improved Image
Captioning via Policy Gradient optimization of SPIDEr,” 2017 IEEE Inter-
national Conference on Computer Vision (ICCV), pp. 873–881, Oct. 2017,
arXiv: 1612.00370. [Online]. Available: http://arxiv.org/abs/1612.00370

#### SPIDEr-max
[7] E. Labbé, T. Pellegrini, and J. Pinquier, “Is my automatic audio captioning system so bad? spider-max: a metric to consider several caption candidates,” Nov. 2022. [Online]. Available: https://hal.archives-ouvertes.fr/hal-03810396

#### FENSE
[8] Z. Zhou, Z. Zhang, X. Xu, Z. Xie, M. Wu, and K. Q. Zhu, Can Audio Captions Be Evaluated with Image Caption Metrics? arXiv, 2022. [Online]. Available: http://arxiv.org/abs/2110.04684 

## Citation
If you use **SPIDEr-max**, you can cite the following paper using BibTex:
```
@inproceedings{labbe:hal-03810396,
    TITLE = {{Is my automatic audio captioning system so bad? spider-max: a metric to consider several caption candidates}},
    AUTHOR = {Labb{\'e}, Etienne and Pellegrini, Thomas and Pinquier, Julien},
    URL = {https://hal.archives-ouvertes.fr/hal-03810396},
    BOOKTITLE = {{Workshop DCASE}},
    ADDRESS = {Nancy, France},
    YEAR = {2022},
    MONTH = Nov,
    KEYWORDS = {audio captioning ; evaluation metric ; beam search ; multiple candidates},
    PDF = {https://hal.archives-ouvertes.fr/hal-03810396/file/Labbe_DCASE2022.pdf},
    HAL_ID = {hal-03810396},
    HAL_VERSION = {v1},
}
```

## Contact
Maintainer:
- Etienne Labbé "Labbeti": labbeti.pub@gmail.com

<!-- # -*- coding: utf-8 -*- -->

<div align="center">

# Audio Captioning metrics (aac-metrics)

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10.1-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
<a href="https://github.com/Labbeti/aac-metrics/actions"><img alt="Build" src="https://img.shields.io/github/workflow/status/Labbeti/aac-metrics/Python%20package%20using%20Pip/main?style=for-the-badge&logo=github"></a>

Audio Captioning metrics source code, designed for Pytorch.

</div>

This package is a tool to evaluate sentences produced by automatic models to caption image or audio.
The results of BLEU [1], ROUGE-L [2], METEOR [3], CIDEr [4], SPICE [5] and SPIDEr [6] are consistents with https://github.com/audio-captioning/caption-evaluation-tools.


## Installation
Install the pip package:
```
pip install https://github.com/Labbeti/aac-metrics
```

Download the external code needed for METEOR, SPICE and PTBTokenizer:
```
aac-metrics-download
```

<!-- ## Why using this package?
- Easy installation with pip
- Consistent with audio caption metrics https://github.com/audio-captioning/caption-evaluation-tools
- Removes code boilerplate inherited from python 2
- Provides functions and classes to compute metrics separately -->

## Examples

### Evaluate all metrics
```python
from aac_metrics import aac_evaluate

candidates = ["a man is speaking", ...]
mult_references = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ...]

global_scores, _ = aac_evaluate(candidates, mult_references)
print(global_scores)
# dict containing the score of each aac metric: "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider_d", "spice", "spider"
# {"bleu_1": tensor(0.7), "bleu_2": ..., ...}
```

### Evaluate a specific metric
```python
from aac_metrics.functional import coco_cider_d

candidates = [...]
mult_references = [[...], ...]

global_scores, local_scores = coco_cider_d(candidates, mult_references)
print(global_scores)
# {"cider_d": tensor(0.1)}
print(local_scores)
# {"cider_d": tensor([0.9, ...])}
```

### Experimental SPIDEr-max metric
```python
from aac_metrics.functional import spider_max

mult_candidates = [[...], ...]
mult_references = [[...], ...]

global_scores, local_scores = spider_max(mult_candidates, mult_references)
print(global_scores)
# {"spider": tensor(0.1)}
print(local_scores)
# {"spider": tensor([0.9, ...])}
```

## Requirements
### Python packages

The requirements are automatically installed when using `pip install` on this repository.
```
torch >= 1.10.1
numpy >= 1.21.2
pyyaml >= 6.0
tqdm >= 4.64.0
```

### External requirements

- `java` >= 1.8 is required to compute METEOR, SPICE and use the PTBTokenizer.
Most of these functions can specify a java executable path with `java_path` argument.

- `unzip` command to extract SPICE zipped files.

## Metrics

### Coco metrics
| Metric | Origin | Range | Short description |
|:---:|:---:|:---:|:---:|
| BLEU [1] | machine translation | [0, 1] | Precision of n-grams |
| ROUGE-L [2] | machine translation | [0, 1] | Longest common subsequence |
| METEOR [3] | machine translation | [0, 1] | Cosine-similarity of frequencies |
| CIDEr [4] | image captioning | [0, 10] | Cosine-similarity of TF-IDF |
| SPICE [5] | image captioning | [0, 1] | FScore of semantic graph |
| SPIDEr [6] | image captioning | [0, 5.5] | Mean of CIDEr and SPICE |

### Other metrics
<!-- TODO : cite workshop paper for SPIDEr-max -->
| Metric | Origin | Range | Short description |
|:---:|:---:|:---:|:---:|
| SPIDEr-max | audio captioning | [0, 5.5] | Max of multiples candidates SPIDEr scores |

## References
[1] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: a
method for automatic evaluation of machine translation,” in Proceed-
ings of the 40th Annual Meeting on Association for Computational
Linguistics - ACL ’02. Philadelphia, Pennsylvania: Association
for Computational Linguistics, 2001, p. 311. [Online]. Available:
http://portal.acm.org/citation.cfm?doid=1073083.1073135

[2] C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,”
in Text Summarization Branches Out. Barcelona, Spain: Association
for Computational Linguistics, Jul. 2004, pp. 74–81. [Online]. Available:
https://aclanthology.org/W04-1013

[3] M. Denkowski and A. Lavie, “Meteor Universal: Language Specific
Translation Evaluation for Any Target Language,” in Proceedings of the
Ninth Workshop on Statistical Machine Translation. Baltimore, Maryland,
USA: Association for Computational Linguistics, 2014, pp. 376–380.
[Online]. Available: http://aclweb.org/anthology/W14-3348

[4] R. Vedantam, C. L. Zitnick, and D. Parikh, “CIDEr: Consensus-based
Image Description Evaluation,” arXiv:1411.5726 [cs], Jun. 2015, arXiv:
1411.5726. [Online]. Available: http://arxiv.org/abs/1411.5726

[5] P. Anderson, B. Fernando, M. Johnson, and S. Gould, “SPICE: Semantic
Propositional Image Caption Evaluation,” arXiv:1607.08822 [cs], Jul. 2016,
arXiv: 1607.08822. [Online]. Available: http://arxiv.org/abs/1607.08822

[6] S. Liu, Z. Zhu, N. Ye, S. Guadarrama, and K. Murphy, “Improved Image
Captioning via Policy Gradient optimization of SPIDEr,” 2017 IEEE Inter-
national Conference on Computer Vision (ICCV), pp. 873–881, Oct. 2017,
arXiv: 1612.00370. [Online]. Available: http://arxiv.org/abs/1612.00370

## Cite the aac-metrics package
The associated paper has been accepted but it will be published after the DCASE2022 workshop.

If you use this code, you can cite with the following **temporary** citation:
<!-- TODO : update citation and create CITATION.cff file -->
```
@inproceedings{Labbe2022,
    author = "Etienne Labbe, Thomas Pellegrini, Julien Pinquier",
    title = "IS MY AUTOMATIC AUDIO CAPTIONING SYSTEM SO BAD? SPIDEr-max: A METRIC TO CONSIDER SEVERAL CAPTION CANDIDATES",
    month = "November",
    year = "2022",
}
```


# Audio Captioning metrics (aac-metrics)

This package is a tool to evaluate sentences produced by automatic models to caption image or audio.
The results of BLEU, ROUGE-L, METEOR, CIDEr, SPICE and SPIDEr are consistents with https://github.com/audio-captioning/caption-evaluation-tools.

## Installation
```
pip install https://github.com/Labbeti/aac-metrics
```

<!-- ## Why using this package?
- Easy installation with pip
- Consistent with audio caption metrics https://github.com/audio-captioning/caption-evaluation-tools
- Removes code boilerplate inherited from python 2
- Provides functions and classes to compute metrics separately -->

## Usage

### Evaluate all metrics
```python
from aac_metrics import aac_evaluate

candidates = ["a man is speaking", ...]
mult_references = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ...]

global_scores, _ = aac_evaluate(candidates, mult_references)
print(global_scores)
# dict containing the score of each metric: "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider_d", "spice", "spider"
# {"bleu_1": tensor(0.7), "bleu_2": ..., ...}
```

### Evaluate specific metrics
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

### Experimental SPIDEr-max
```python
from aac_metrics.experimental.spider_max import spider_max

mult_candidates = [[...], ...]
mult_references = [[...], ...]

global_scores, local_scores = spider_max(mult_candidates, mult_references)
print(global_scores)
# {"spider": tensor(0.1)}
print(local_scores)
# {"spider": tensor([0.9, ...])}
```

<!-- ## References
TODO -->

# Change log

All notable changes to this project will be documented in this file.

## [0.4.1] 2023-04-13
### Deleted
- Old unused files `package_tree.rst`, `fluency_error.py`, `sbert.py` and `spider_err.py`.

## [0.4.0] 2023-04-13
### Added
- Argument `return_probs` for fluency error metric.

### Changed
- Rename `SPIDErErr` to `SPIDErFL` to match DCASE2023 metric name.
- Rename `SBERT` to `SBERTSim` to avoid confusion with SBERT model name.
- Rename `FluencyError` to `FluErr`.
- Check if Java executable version between 8 and 11.

### Fixed
- `SPIDErFL` sentences scores outputs when using `return_all_scores=True`.
- Argument `reset_state` in `SPIDErFL`, `SBERTSim`, `FluErr` and `FENSE` when using their functional interface.
- Classes and functions factories now support SPICE and CIDEr-D metrics.
- `SBERTSim` class instantiation.

## [0.3.0] 2023-02-27
### Added
- Parameters `timeout` and `separate_cache_dir` in `SPICE` function and class.
- Documentation pages with sphinx.
- Parameter `language` in `METEOR` function and class.
- Options to download only `PTBTokenizer`, `METEOR`, `SPICE` or `FENSE` in `download.py`.
- `SBERT` and `FluencyError` metrics extracted from `FENSE`.
- `SPIDErErr` metric which combines `SPIDEr` with `FluencyError`.
- Parameter `reset_state` in `SBERT`, `FluencyError`, `SPIDErErr` and `FENSE` functions and classes.

### Changed
- Fix README typo and SPIDEr-max tables.

### Fixed
- Workflow badge with Github changes. (https://github.com/badges/shields/issues/8671)

## [0.2.0] 2022-12-14
### Added
- `FENSE` class and function metric, with fluency error rate and raw output probabilities.
- Unittest with `fense` repository.
- `load_metric` function in init to match huggingface evaluation package.

### Changed
- Rename `global_scores` to `corpus_scores` and `local_scores` to `sents_scores`.
- Rename `CustomEvaluate` to `Evaluate` and `custom_evaluate` to `evaluate`.
- Set default cache path to `$HOME/.cache`.
- Remove 'coco' prefix to file, functions and classes names to have cleaner names.

### Fixed
- `FENSE` metric error when computing scores with less than `batch_size` sentences.

## [0.1.2] 2022-10-31
### Added
- All candidates scores option `return_all_cands_scores` for SPIDEr-max.
- Functions `is_mono_sents` and `is_mult_sents` to detect `list[str]` sentences and `list[list[str]]` multiples sentences.
- Functions `flat_list` and `unflat_list` to flat multiples sentences to sentences.

### Changed
- Update default value used for `return_all_scores` in cider and rouge functions.
- Update internal metric factory with functions instead of classes to avoid cyclic dependency.

### Fixed
- Fix SPIDEr-max local scores output shape.

## [0.1.1] 2022-09-30
### Added
- Documentation for metric functions and classes.
- A second larger example for unit testing.

### Changed
- Update README information, references and description.

### Fixed
- SPIDEr-max computation with correct global and local outputs.
- Unit testing for computing SPICE metric from caption-evaluation-tools.

## [0.1.0] 2022-09-28
### Added
- BLEU, METEOR, ROUGE-l, SPICE, CIDEr and SPIDEr metrics functions and modules.
- SPIDEr-max experimental implementation.
- Installation script in download.py.
- Evaluation script in evaluate.py.
- Unittest with `caption-evaluation-tools` repository.

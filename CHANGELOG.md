# Change log

All notable changes to this project will be documented in this file.

## [0.6.0] 2025-06-29
### Changed
- Constants are replaced by `typing.get_args` from Literal type.
- Utilities now uses `pythonwrench`package.

### Fixed
- Java version parsing. [PR #13](https://github.com/Labbeti/aac-metrics/pull/13)
- SPICE URL with https protocol. [PR #13](https://github.com/Labbeti/aac-metrics/pull/13)

## [0.5.5] 2025-01-20
### Added
- New `CLAPSim` metric based on the embeddings given by CLAP model.
- New `MACE` metric based on `CLAPSim` and `FER` metrics.
- DCASE2024 challenge metric set, class and functions.
- Preprocess option in `evaluate` now accepts custom callable value.
- List of bibtex sources in `data/papers.bib` file.

### Changed
- Improve metric output typing for language servers with typed dicts.
- `batch_size` can now be `None` to take all inputs at once into the model.

### Fixed
- `bert_score` option in download script.

## [0.5.4] 2024-03-04
### Fixed
- Backward compatibility of `BERTScoreMrefs` with torchmetrics prior to 1.0.0.

### Deleted
- `Version` class to use `packaging.version.Version` instead.

## [0.5.3] 2024-01-09
### Fixed
- Fix `BERTScoreMrefs` computation when all multiple references sizes are equal.
- Check for empty timeout list in `SPICE` metric.

## [0.5.2] 2024-01-05
### Changed
- `aac-metrics` is now compatible with `transformers>=4.31`.
- Rename default device value `"auto"` to `"cuda_if_available"`.

## [0.5.1] 2023-12-20
### Added
- Check sentences inputs for all metrics.

### Fixed
- Fix `BERTScoreMRefs` metric with 1 candidate and 1 reference.

## [0.5.0] 2023-12-08
### Added
- New `Vocab` metric to compute vocabulary size and vocabulary ratio.
- New `BERTScoreMRefs` metric wrapper to compute BERTScore with multiple references.

### Changed
- Rename metric `FluErr` to `FER`.

### Fixed
- `METEOR` localization issue. ([#9](https://github.com/Labbeti/aac-metrics/issues/9))
- `SPIDErMax` output when `return_all_scores=False`.

## [0.4.6] 2023-10-10
### Added
- Argument `clean_archives` for `SPICE` download.

### Changed
- Check if newline character is in the sentences before ptb tokenization. ([#6](https://github.com/Labbeti/aac-metrics/issues/6))
- `SPICE` no longer requires bash script files for installation.

### Fixed
- Maximal version of `transformers` dependancy set to 4.31.0 to avoid error with `FENSE` and `FluErr` metrics.
- `SPICE` crash message and error output files.
- Default value for `Evaluate` `metrics` argument.

### Deleted
- Remove now useless `use_shell` option for download.

## [0.4.5] 2023-09-12
### Added
- Argument `use_shell` for `METEOR` and `SPICE` metrics and `download` function to fix Windows-OS specific error.

### Changed
- Rename `evaluate.py` script to `eval.py`.

### Fixed
- Workflow on main branch.
- Examples in README and doc with at least 2 sentences, and add a warning on all metrics that requires at least 2 candidates.

## [0.4.4] 2023-08-14
### Added
- `Evaluate` class now implements a `__hash__` and `tolist()` methods.
- `BLEU` 1 to n classes and functions.
- Get and set global user paths for cache, java and tmp.

### Changed
- Function `get_install_info` now returns `package_path`.
- `AACMetric` now indicate the output type when using `__call__` method.
- Rename `AACEvaluate` to `DCASE2023Evaluate` and use `dcase2023` metric set instead of `all` metric set.

### Fixed
- `sbert_sim` name in internal instantiation functions.
- Path management for Windows.

## [0.4.3] 2023-06-15
### Changed
- `AACMetric` is no longer a subclass of `torchmetrics.Metric` even when it is installed. It avoid dependency to this package and remove potential errors due to Metric base class.
- Java 12 and 13 are now allowed in this package.

### Fixed
- Output name `sbert_sim` in FENSE and SBERTSim classes.
- `Evaluate` class instantiation with `torchmetrics` >= 0.11.
- `evaluate.py` script when using a verbose mode != 0.

## [0.4.2] 2023-04-19
### Fixed
- File `install_spice.sh` is now in `src/aac_metrics` directory to fix download from a pip installation. ([#3](https://github.com/Labbeti/aac-metrics/issues/3))
- Java version retriever to avoid exception when java version is correct. ([#2](https://github.com/Labbeti/aac-metrics/issues/2))

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
- Check if Java executable version between 8 and 11. ([#1](https://github.com/Labbeti/aac-metrics/issues/1))

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

# Change log

All notable changes to this project will be documented in this file.

## [0.1.2] UNRELEASED
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

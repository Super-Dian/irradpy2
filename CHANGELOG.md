# Changelog

All notable changes to `irradpy2` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.10] — 2026-08-18

### Changed

- Packaging metadata updated to `pyproject.toml` (setuptools ≥ 77) with SPDX
  license expression, keywords, and classifiers.
- `csd` subpackage now wraps all detection methods with a `RuntimeWarning`
  filter so numerical edge cases do not pollute stdout (results unchanged).

### Added

- User documentation under `docs/` (installation, getting started, user guide,
  API reference).
- `readme.md` expanded with documentation index and quick-start example.

## [1.0.x] — earlier releases

Earlier release history was not recorded in this file. Highlights of the
current feature set (all present as of 1.0.10):

- 62 clear-sky irradiance models (`irradpy2.clear_sky`)
- 23 clear-sky detection methods (`irradpy2.csd`)
- 7 irradiance separation models (`irradpy2.separation`)
- 5 GHI forecasting methods including RNN/LSTM/Informer (`irradpy2.forecast`)
- 6 ground-station downloaders (`irradpy2.getirrad`)
- 3 gridded/reanalysis downloaders with point/area modes (`irradpy2.getgridded`)
- Evaluation metrics and plotting utilities (`irradpy2.tool`)

[1.0.10]: dist/irradpy2-1.0.10.tar.gz

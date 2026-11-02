# Installation

## Requirements

- Python **3.10 or newer** (`requires-python = ">=3.10"`)
- Runtime dependencies (installed automatically):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `requests`
  - `pytz`
  - `scikit-learn`
  - `torch`

`torch` is required only by the deep-learning forecasting methods
(`forecast_rnn`, `forecast_lstm`, `forecast_informer`), but it is declared as a
core dependency. If you need a lighter environment for data acquisition and
clear-sky modeling only, install the other dependencies manually and avoid the
`forecast` subpackage.

## Install from PyPI

Once the package is published on PyPI:

```bash
pip install irradpy2
```

## Install from a built distribution

This repository ships pre-built artifacts in `dist/`:

```bash
pip install dist/irradpy2-1.0.10-py3-none-any.whl
```

or from the source distribution:

```bash
pip install dist/irradpy2-1.0.10.tar.gz
```

## Install from source (development)

```bash
git clone <repository-url>
cd irradpy2_project
pip install -e .
```

Editable install keeps the package linked to your working copy.

## Build from source

The project uses a standard `pyproject.toml` (setuptools backend):

```bash
python -m pip install build
python -m build
```

This produces a wheel and sdist under `dist/`.

## Verify the installation

```python
import irradpy2
print(irradpy2.__version__)   # e.g. 1.0.10

from irradpy2.clear_sky import tj
from irradpy2.csd import Ellis2018CSD
from irradpy2.separation import engerer2
from irradpy2.forecast import forecast_persistence
from irradpy2.tool import cal_metrics
```

## Platform notes

- The package is platform-independent pure Python (`py3-none-any` wheel).
- Downloaders require network access; some require credentials (see the
  [credentials table](index.md#external-data-credentials)).
- Forecasting models run on CPU by default; no GPU is required.

## Known packaging notes

- `clear_sky/mrmv6` reads an external lookup file `AEROSOL-TABLE.txt` from the
  **current working directory** (`np.loadtxt("AEROSOL-TABLE.txt", ...)`), not
  from the installed package. Copy `src/irradpy2/clear_sky/AEROSOL-TABLE.txt`
  into your working directory before calling `mrmv6`.
- Neither `AEROSOL-TABLE.txt` nor `adelaide_airport_inputs_sample.csv` is
  currently included in the built wheel or sdist (`dist/`). When installing
  from `dist/`, obtain these files from the source tree if you need them.

# irradpy2

[![PyPI](https://img.shields.io/pypi/v/irradpy2.svg)](https://pypi.org/project/irradpy2/)
[![Python versions](https://img.shields.io/pypi/pyversions/irradpy2.svg)](https://pypi.org/project/irradpy2/)
[![License: MIT](https://img.shields.io/github/license/Super-Dian/irradpy2.svg)](https://github.com/Super-Dian/irradpy2/blob/main/LICENSE)

`irradpy2` is a Python package for solar irradiance data acquisition and
analysis, covering clear-sky modeling and detection, irradiance separation,
GHI forecasting, and evaluation.

## Modules

| Module | Description |
|---|---|
| `getirrad` | Ground observations from BSRN, MIDC, SOLRAD, SRML, and SURFRAD |
| `getgridded` | Point and area data from MERRA-2, ERA5, and CAMS |
| `clear_sky` | 92 Clear-sky irradiance models |
| `csd` | 23 clear-sky detection methods |
| `separation` | 7 irradiance separation models |
| `forecast` | Persistence, ARIMA, RNN, LSTM, and Informer forecasting |
| `tool.py` | Data preparation, solar geometry, evaluation metrics, and plotting |

## Installation

Python 3.10 or newer is required.

```bash
pip install irradpy2
```

## Common units

- Irradiance — W/m²
- Temperature — °C
- Pressure — hPa
- Relative humidity — %
- Wind speed and direction — m/s and °
- Precipitation and snow — mm (mm/h for rates)

## Examples

```bash
python -m irradpy2.example.ex_download_basic
python -m irradpy2.example.ex_clearsky_csd
python -m irradpy2.example.ex_separation_forecast
```

## Function documentation

Model descriptions, references, and parameter information are provided
in the source files. Function docstrings can also be viewed with
Python's `help()`:

```python
from irradpy2.separation import engerer2

help(engerer2)
```
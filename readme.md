# irradpy2

`irradpy2` is a Python package for solar irradiance data acquisition, processing, modeling, separation, forecasting, and evaluation.

The package provides a consistent interface for working with ground-based observations, gridded meteorological data, clear-sky irradiance models, clear-sky detection methods, irradiance separation models, and forecasting methods.

## Features

- Download ground-based solar irradiance observations (BSRN, MIDC, SAURAN, SOLRAD, SRML, SURFRAD)
- Download gridded and reanalysis meteorological data (MERRA-2, ERA5, CAMS)
- Prepare multi-source data for irradiance analysis
- Run 62 clear-sky irradiance models
- Detect clear-sky periods with 23 clear-sky detection (CSD) methods
- Estimate diffuse and direct irradiance components with 7 separation models
- Forecast global horizontal irradiance (persistence, ARIMA, RNN, LSTM, Informer)
- Calculate common evaluation metrics (MSE, RMSE, MAE, MBE, R²)
- Visualize model and observation results

## Installation

`irradpy2` requires Python 3.10 or newer.

From PyPI (once published):

```bash
pip install irradpy2
```

From a built wheel or source distribution:

```bash
pip install dist/irradpy2-1.0.10-py3-none-any.whl
# or
pip install dist/irradpy2-1.0.10.tar.gz
```

From a local source checkout:

```bash
pip install -e .
```

Runtime dependencies: `numpy`, `pandas`, `matplotlib`, `requests`, `pytz`, `scikit-learn`, `torch`.

## Quick start

```python
import numpy as np
from irradpy2.tool import read_csv, cal_zen, cal_metrics
from irradpy2.clear_sky import tj
from irradpy2.csd import Ellis2018CSD

data = read_csv("irradiance.csv")
data = cal_zen(data, latitude=39.742, longitude=-105.18, time_col="UTC", zen_col="zen")

# clear-sky model inputs use zenith angle in radians
sza = np.radians(data["zen"].to_numpy())
ghi_clear, dni_clear, dhi_clear = tj(sza, data["UTC"].to_numpy())

# clear-sky detection flag: 1 = clouds suspected, 0 = clear
csd = Ellis2018CSD(data["ghi"].to_numpy(), ghi_clear)
```

See [docs](docs/index.md) for the full user guide and API reference.

## Documentation

| Document | Content |
|---|---|
| [Documentation home](docs/index.md) | Table of contents |
| [Installation](docs/installation.md) | Install and build instructions |
| [Getting started](docs/getting-started.md) | End-to-end walkthrough |
| [Data acquisition](docs/user-guide/data-acquisition.md) | `getirrad` and `getgridded` downloaders |
| [Clear-sky models](docs/user-guide/clear-sky-models.md) | 62 clear-sky irradiance models |
| [Clear-sky detection](docs/user-guide/clear-sky-detection.md) | 23 CSD methods |
| [Irradiance separation](docs/user-guide/separation.md) | 7 separation models |
| [Forecasting](docs/user-guide/forecasting.md) | 5 GHI forecasting methods |
| [Data tools](docs/user-guide/data-tools.md) | Time series, metrics, and plotting utilities |
| [Data conventions](docs/user-guide/data-conventions.md) | Variable names, units, and return ordering |
| [API reference: clear_sky](docs/reference/clear-sky.md) | One entry per clear-sky model (62) |
| [API reference: csd](docs/reference/csd.md) | One entry per detection method (23) |
| [API reference: separation](docs/reference/separation.md) | One entry per separation model (7) |
| [API reference: forecast](docs/reference/forecast.md) | One entry per forecasting method (5) |
| [API reference: getirrad](docs/reference/getirrad.md) | Ground-station downloaders |
| [API reference: getgridded](docs/reference/getgridded.md) | Reanalysis downloaders |
| [API reference: tool](docs/reference/tool.md) | Data, evaluation, and plotting utilities |
| [Changelog](CHANGELOG.md) | Release history |

## Package structure

```text
irradpy2
├── clear_sky    # Clear-sky irradiance models (62)
├── csd          # Clear-sky detection methods (23)
├── separation   # Irradiance separation models (7)
├── forecast     # GHI forecasting methods (5)
├── getirrad     # Ground-based observation downloaders (6)
├── getgridded   # Gridded/reanalysis downloaders (3)
└── tool.py      # Data processing, evaluation, and visualization utilities
```

## Data conventions

| Variable | Meaning | Typical unit |
|---|---|---|
| `UTC` | UTC timestamp | datetime |
| `ghi` | Global horizontal irradiance | W m⁻² |
| `dni` | Direct normal irradiance | W m⁻² |
| `dhi` | Diffuse horizontal irradiance | W m⁻² |
| `zen` | Solar zenith angle | degree |
| `sza` | Solar zenith angle used by clear-sky models | **radian** |
| `press` | Atmospheric pressure | hPa or mb |
| `wv` | Precipitable water | model-dependent |
| `aod700` / `aod550` | Aerosol optical depth | dimensionless |
| `albedo` | Surface albedo | dimensionless |

Full details, including clear-sky model return ordering, are in
[Data conventions](docs/user-guide/data-conventions.md).

## Citation

If you use `irradpy2` in published work, please cite the package version
(see [CITATION.cff](CITATION.cff)).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2024 Peiyu Lin.

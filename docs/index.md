# irradpy2 documentation

`irradpy2` is a Python package for solar irradiance data acquisition, processing,
modeling, separation, forecasting, and evaluation.

## Getting started

New users should read these pages in order:

1. [Installation](installation.md) — how to install the package and its dependencies.
2. [Getting started](getting-started.md) — an end-to-end walkthrough from raw data to evaluation.

## User guide

| Guide | Content |
|---|---|
| [Data acquisition](user-guide/data-acquisition.md) | Ground-based (`getirrad`) and gridded (`getgridded`) downloaders |
| [Clear-sky models](user-guide/clear-sky-models.md) | 62 clear-sky irradiance models and their inputs |
| [Clear-sky detection](user-guide/clear-sky-detection.md) | 23 clear-sky detection (CSD) methods |
| [Irradiance separation](user-guide/separation.md) | 7 models that split GHI into diffuse and direct components |
| [Forecasting](user-guide/forecasting.md) | Persistence, ARIMA, RNN, LSTM, and Informer GHI forecasting |
| [Data tools](user-guide/data-tools.md) | Time-series utilities, evaluation metrics, and plotting |
| [Data conventions](user-guide/data-conventions.md) | Variable names, units, and return ordering |

## Reference

Per-function API documentation — signature, parameters, returns, and
literature references, one entry per public function:

- [`irradpy2.clear_sky`](reference/clear-sky.md) — 62 clear-sky irradiance models
- [`irradpy2.csd`](reference/csd.md) — 23 clear-sky detection methods
- [`irradpy2.separation`](reference/separation.md) — 7 irradiance separation models
- [`irradpy2.forecast`](reference/forecast.md) — 5 GHI forecasting methods
- [`irradpy2.getirrad`](reference/getirrad.md) — ground-station downloaders
- [`irradpy2.getgridded`](reference/getgridded.md) — reanalysis downloaders
- [`irradpy2.tool`](reference/tool.md) — data, evaluation, and plotting utilities
- [Changelog](../CHANGELOG.md) — release history.

## Package layout

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

## External data credentials

Some downloaders require accounts or API keys with external services:

| Source | Credential | Environment service |
|---|---|---|
| BSRN | username / password | [BSRN](https://bsrn.awi.de/) station account |
| MERRA-2 | username / password | [NASA Earthdata](https://urs.earthdata.nasa.gov/) |
| ERA5 | api_url / api_key | [Copernicus CDS](https://cds.climate.copernicus.eu/) |
| CAMS | api_url / api_key | [Copernicus ADS](https://atmosphere.copernicus.eu/) |

The remaining sources (MIDC, SAURAN, SOLRAD, SRML, SURFRAD) require no credentials.

## Contributing

Bug reports and feature requests are welcome via the project issue tracker.
When reporting a problem, include the package version (`irradpy2.__version__`),
a minimal reproducible example, and the data source involved.

## License

MIT — see [LICENSE](../LICENSE).

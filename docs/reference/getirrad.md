# `irradpy2.getirrad` API reference

One entry per public function. Function names are listed in the module
overview; parameter semantics follow the source docstrings.

Reference lists are reproduced verbatim from the source docstrings.
They attribute the published formulation each model implements;
they are implementation provenance, not verified bibliographic records.

Functions (6): `download_bsrn`, `download_midc`, `download_sauran`, `download_solrad`, `download_srml`, `download_surfrad`

---

### `download_bsrn(site, start, end, username, password, save_path)`

Download BSRN shortwave radiation data and save the result as a CSV file.

**Parameters**

- `site` (*str*) — BSRN station identifier, for example ``"cab"`` or ``"qiq"``.
- `start` (*str*) — Start date in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
- `end` (*str*) — End date in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
- `username` (*str*) — BSRN FTP username.
- `password` (*str*) — BSRN FTP password.
- `save_path` (*str*) — Absolute output CSV file path.

**References**

BSRN homepage:
https://bsrn.awi.de/

BSRN FTP data retrieval:
https://bsrn.awi.de/data/data-retrieval-via-ftp/

### `download_midc(site, start, end, save_path)`

Public high-level MIDC download interface.

**Parameters**

- `site` (*str*) — MIDC site ID, for example "BMS".
- `start` (*str*) — Start date in YYYYMMDD format.
- `end` (*str*) — End date in YYYYMMDD format.
- `save_path` (*str*) — Output CSV file path.

### `download_sauran(site, start, end, save_path)`

Download SAURAN data and save the processed result as a CSV file.

**Parameters**

- `site` (*str*) — SAURAN station identifier, for example "NMU" or "UNV".
- `start` (*str*) — Start date in YYYYMMDD or YYYY-MM-DD format.
- `end` (*str*) — End date in YYYYMMDD or YYYY-MM-DD format.
- `save_path` (*str*) — Absolute output CSV file path.

**Returns**

None     The processed data are written directly to save_path.

### `download_solrad(site, start, end, save_path)`

Download SOLRAD data for a given site and save the processed result as a CSV file.  This is the public high-level SOLRAD download interface.

**Parameters**

- `site` (*str*) — SOLRAD station identifier. See the NOAA SOLRAD page for available station information: https://gml.noaa.gov/grad/solrad/
- `start` (*str*) — Start date in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
- `end` (*str*) — End date in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
- `save_path` (*str*) — Absolute output CSV file path.

**References**

NOAA SOLRAD homepage:
https://gml.noaa.gov/grad/solrad/

NOAA SOLRAD data access:
https://gml.noaa.gov/aftp/data/radiation/solrad/

### `download_srml(site, start, end, save_path)`

Download SRML data for a given site and save the processed result as a CSV file.  This is the public high-level SRML download interface.

**Parameters**

- `site` (*str*) — SRML site code (for example, ``"BUO"``).
- `start` (*str*) — Start date in ``YYYYMMDD`` format.
- `end` (*str*) — End date in ``YYYYMMDD`` format.
- `save_path` (*str*) — Absolute output CSV file path.

**References**

SRML homepage:
http://solardata.uoregon.edu/

SRML data archive:
http://solardata.uoregon.edu/download/archive/

### `download_surfrad(site, start, end, save_path)`

Public SURFRAD download function (replaces fetch_surfrad_data).

**Parameters**

- `site` (*str*) — SURFRAD site ID (e.g., "bon", "dra", "fpk").
- `start` (*str*) — Start date (YYYY-MM-DD or YYYYMMDD).
- `end` (*str*) — End date (YYYY-MM-DD or YYYYMMDD).
- `save_path` (*str, optional*) — Output CSV path. If None, the filename is auto-generated.


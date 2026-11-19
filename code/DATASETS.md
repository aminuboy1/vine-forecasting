# Datasets

All URLs below were checked on 2026-11-12/13 from the laptop (HTTP 200 and/or a successful, checksum-verified download).
Every download script pins sizes and checksums and refuses files that do not match. Raw data is **never committed**
(`.gitignore` excludes `code/data/`). Set `TSL_DATA=/path` to keep data outside the repository (recommended on servers).
Status **D** = downloaded and checksum-verified.

## Standard long-term forecasting collection (the paper's streams)

Source: official thuml Hugging Face mirror `thuml/Time-Series-Library`, **pinned to commit
`2b66e59ee19dac8f6f19fb5d4997f289fdfea357`** (`https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/<commit>/<path>`).
Script: `scripts/download_tslib.py` (stdlib only). Verified by size + git blob SHA-1 (non-LFS) or LFS SHA-256, plus SHA-256.
ETT files from the original repo `https://github.com/zhouhaoyi/ETDataset` (`--ett-source github`) have the same git blob
SHAs, i.e. they are byte-identical.

| Dataset | File | Rows x Cols | Bytes | SHA-256 | Licence | Status |
|---|---|---|---|---|---|---|
| ETTh1 | ETT-small/ETTh1.csv | 17,420 x 7 | 2,589,657 | f18de3ad...6ee066 | CC BY-ND 4.0 (ETDataset `LICENSE`, verified) | D |
| ETTh2 | ETT-small/ETTh2.csv | 17,420 x 7 | 2,417,960 | a3dc2c59...9218b | CC BY-ND 4.0 | D |
| ETTm1 | ETT-small/ETTm1.csv | 69,680 x 7 | 10,360,719 | 6ce1759b...986c9e | CC BY-ND 4.0 | D |
| ETTm2 | ETT-small/ETTm2.csv | 69,680 x 7 | 9,677,236 | db973ca2...add1fd | CC BY-ND 4.0 | D |
| ECL | electricity/electricity.csv | 26,304 x 321 | 95,581,762 | 7e45845d...4497ad | derived from UCI ElectricityLoadDiagrams (CC BY 4.0) | D |
| Traffic | traffic/traffic.csv | 17,544 x 862 | 136,478,119 | cb06463d...dab0b16 | Caltrans PeMS-derived; no explicit licence | D |
| Weather | weather/weather.csv | 52,696 x 21 | 7,235,425 | 34ee981d...ba63 | Max-Planck BGC Jena station data; no explicit licence | D |
| Exchange | exchange_rate/exchange_rate.csv | 7,588 x 8 | 637,800 | 48b4d9d3...89842 | from LSTNet (Lai et al. 2018); no explicit licence | D |

The HF dataset card declares CC-BY-4.0 for the bundle; the original sources' terms take precedence. Weather contains
`-9999` missing-value codes in some rows (kept raw on purpose: the DRiC "guard" stress case).

## Optional extra stream (not used in the paper)

| Dataset | Source (verified) | Size | Checksum | Licence | Status |
|---|---|---|---|---|---|
| OPSD time_series 60-min, v2020-10-06 | `https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv` (+ `datapackage.json`) | 130,339,665 B (50,401 rows x 300 cols, 2015-01-01 .. 2020-09-30 UTC) | sha256 6a7f2bc5...04c8 (equals the hash in datapackage.json) | **Not stated**: datapackage.json has no `licenses` field; attribution text requested (DOI 10.25832/time_series/2020-10-06); primary data = ENTSO-E Transparency terms. **Verify before publication.** | D |

Loader: `OPSD-DE` = national load + 4 control areas (50Hertz, Amprion, TenneT, TransnetBW), 5 channels;
`OPSD-IT` = national load (1 channel). `with_tso_forecast=True` appends the TSO day-ahead forecast column (appendix covariate).
Script: `scripts/download_opsd.py`.

## Splits and scaling (implemented in `tsl/data/splits.py`)

* `fsnet` 20:5:75 (FSNet/OneNet/Proceed; ratios of the full length), `audit` 25:5:70, `standard`
  (ETT 12/4/4 months, others 70/10/20), `fsnet_ett_months` (4/1/15 months for ETT; check which variant a compared paper
  used), `ratio`. The z-score scaler is always fitted on the training rows only.

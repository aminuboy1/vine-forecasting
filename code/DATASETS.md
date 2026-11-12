# Datasets

All URLs below were checked on 2026-11-12/13 from the laptop (HTTP 200 and/or a successful, checksum-verified download).
Every download script pins sizes and checksums and refuses files that do not match. Raw data is **never committed**
(`.gitignore` excludes `code/data/`). Set `TSL_DATA=/path` to keep data outside the repository (recommended on servers).

Status on the laptop: **D** = downloaded and verified locally; **S** = scripted only (fetch on the server or later);
**P** = partially downloaded.

## Standard long-term forecasting collection (paper 2 headline, both harnesses)

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

## Paper 2 extra stream

| Dataset | Source (verified) | Size | Checksum | Licence | Status |
|---|---|---|---|---|---|
| OPSD time_series 60-min, v2020-10-06 | `https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv` (+ `datapackage.json`) | 130,339,665 B (50,401 rows x 300 cols, 2015-01-01 .. 2020-09-30 UTC) | sha256 6a7f2bc5...04c8 (equals the hash in datapackage.json) | **Not stated**: datapackage.json has no `licenses` field; attribution text requested (DOI 10.25832/time_series/2020-10-06); primary data = ENTSO-E Transparency terms. **Verify before publication.** | D |

Loader: `OPSD-DE` = national load + 4 control areas (50Hertz, Amprion, TenneT, TransnetBW), 5 channels;
`OPSD-IT` = national load (1 channel). `with_tso_forecast=True` appends the TSO day-ahead forecast column (appendix covariate).
Script: `scripts/download_opsd.py`.

## Paper 1 continual streams

| Dataset | Source (verified) | Content (from inspection) | Size | Licence | Status |
|---|---|---|---|---|---|
| Energy-Stream | EAC Google Drive folder `https://drive.google.com/drive/folders/1OiMLuFBdc56CLekileRjH0xyhDWuoC6C` (`ENERGY-Wind/`), linked from `https://github.com/Onedean/EAC` | 4 periods (`0..3`), N = 103, 113, 122, 134 nodes, T = 8,640 each (10-min sampling inferred) | 33 MB | EAC code Apache-2.0; **no data licence stated** (Baidu KDD Cup 2022 SDWPF origin) | D |
| PEMS-Stream (PEMS3) | same folder (`PEMS/`) | years 2011..2017, N = 655, 715, 786, 822, 834, 850, 871; T = 15,264 per year (5-min) | 715 MB | no licence stated (Caltrans PeMS) | **P**: all 7 graphs + RawData/2011 only. Google Drive started refusing automated downloads ("too many accesses"); re-run `python scripts/download_streams.py --only pems` later or fetch in a browser |
| Air-Stream | same folder (`AIR/`) | years 2016..2019, N = 1087, 1154, 1193, 1202; T ~ 8.4-8.6k (hourly, dropped rows) | 359 MB | no licence stated (quotsoft.net air quality) | D |
| CA-Stream | STBP Drive folder `https://drive.google.com/drive/folders/121ZB-C4ODMKPIofGAMH0EAfSFsmGmEKE` (linked from `https://github.com/Aoyu-Liu/STBP`) | 4 periods, N = 480, 691, 1175, 1698; T = 8928, 8064, 8928, 8640 (5-min, monthly) | 320 MB | no licence stated | D |
| TrafficStream raw (district3F11T17.zip) | Drive file `https://drive.google.com/file/d/1P5wowSaNSWBNCK3mQwESp-G2zsutXc5S/view` (from `https://github.com/AprLie/TrafficStream`) | raw PeMS district 3, 2011-2017 | 2,115,184,578 B (Content-Length) | no licence stated | **S** (blocked by the same Drive refusal; `--only trafficstream --extract`) |
| ECL-Churn (UCI ElectricityLoadDiagrams2011-2014, raw 370 clients) | `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip` (from the UCI dataset page 321) | 140,256 x 370 at 15 min (2011-01-01 00:15 .. 2015-01-01); loader aggregates to hourly (35,064 rows). 212 clients are born after the start (leading zeros), only 2 natural retirements -> use the controlled-retirement schedule | zip 261,335,609 B (sha256 f6c4d0e0...d4e1); txt 710,998,915 B (sha256 d51565f2...124b) | **CC BY 4.0** (UCI page), DOI 10.24432/C58C86 | D |
| CESNET-TimeSeries24 | Zenodo record 13382427 (concept 13382426), `https://zenodo.org/api/records/13382427` | institutions (283 series) and institution_subnets (548), 10-min / 1-h / 1-day aggregates, 19 metrics; 40 weeks (2023-10-09 .. 2024-07-14) | institutions.tar.gz 479,428,489 B; institution_subnets.tar.gz 774,004,934 B (md5 pinned from Zenodo); extracted 3.2 GB | **CC BY 4.0** (Zenodo) | D (aggregated levels). `ip_addresses_sample` (171 MB) optional; `ip_addresses_full` (40 GB) **not** downloaded |
| SynthRegimes | generated (`tsl/data/synth.py`) | Poisson births/deaths, 6 recurring regimes with known spectra | - | ours | generated |

Format of the EAC/STBP npz files (all streams): one file per period, `RawData/<period>.npz` and `graph/<period>_adj.npz`, each
holding a single dense float64 array under key `x`: data `(T, N_period)` with no timestamps; adjacency `(N, N)`. Node
expansion is append-only (period p+1's first N_p nodes are period p's nodes in the same order; checked on the adjacency
prefixes). Sampling intervals are inferred from row counts, not stored; the loader builds synthetic contiguous timestamps.

## Not downloaded / not verified

* **Air-Stream / CA-Stream at full scale** and **EvoXXLTraffic**: Air and CA are downloaded but STGNN baselines on them are
  full-tier work. EvoXXLTraffic's data location is not stated in its repository - not verified, no script.
* **STEV expanding-variate sets** (EElectricity/EPeMS/EWeather, `https://github.com/mb-Ma/STEV`): not checked in this session.
* **PEMS-Few / PEMS-Universal** (same EAC Drive folder) have file-by-file identical sizes to PEMS; identity not verified,
  not downloaded by default (`--only pems_few pems_universal`).
* **Google Drive rate limiting**: `gdown` was refused after ~1.4 GB. The scripts retry with back-off; if it persists,
  download the missing files in a browser into `code/data/eac/PEMS/RawData/` - the script then verifies sizes on re-run.

## Splits and scaling (implemented in `tsl/data/splits.py`, `tsl/continual/stream.py`)

* Online (paper 2): `fsnet` 20:5:75 (FSNet/OneNet/Proceed; ratios of the full length), `audit` 25:5:70, `standard`
  (ETT 12/4/4 months, others 70/10/20), `fsnet_ett_months` (4/1/15 months for ETT; check which variant a compared paper
  used), `ratio`. The z-score scaler is always fitted on the training rows only.
* Continual (paper 1): fixed evaluation partition (count, fixed size, calendar, or the source's own periods), each segment
  split train | gap | val | gap | test (default 6:2:2, gap 0; set `gap: L+H` for the strict checklist). Each variable is
  standardised with the training rows of the first segment where it is available. Val/test rows never reach training,
  scaling or memory under any taskification.

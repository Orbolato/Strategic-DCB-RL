# Data package

This directory contains the conventional-ATM input tables used by the strategic DCB experiments, together with supporting public-source and derived data. The UAM experiment inputs are stored one level above (`../data_uam.csv`, `../arrival_data_uam.csv`, and `../new_taf_list_uam.csv`).

For the full provenance boundary, see [`../../PROVENANCE.md`](../../PROVENANCE.md).

## Experiment-ready inputs

| Path | Role | Source / lineage |
|---|---|---|
| `data.csv` | 15-minute SBGR operational/weather dataset used by the conventional-ATM experiment | Archived processed table from the earlier runway-capacity study; ANAC SIROS/VRA, REDEMET METAR, and aggregate historical Flightradar24-derived quantities were used upstream |
| `new_taf_list.csv` | TAF scenarios decoded to the 15-minute experiment grid | REDEMET/DECEA TAF preprocessing |
| `std_taf.csv` | Standardized TAF table used by the experiment | REDEMET/DECEA TAF preprocessing |
| `curves/*_RCCE_*.csv` | Runway configuration capacity envelopes | Piecewise-linear quantile-regression outputs from the retained R/AMPL workflow |
| `arrival_data.csv` | Flight-level SBGR schedule table feeding the arrival-time correction pipeline | ANAC public SIROS/VRA lineage; no flight-level Flightradar24 row is distributed |
| `VRA_2022.csv`, `VRA_2023.csv`, `VRA_2024.csv` | Historical observed/scheduled flight-time data used to estimate schedule-buffer corrections | ANAC VRA public data |
| `CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv` | Traffic-management-measure context | CGNA ATFM measure records; not a flight movement database |

## Flightradar24 boundary

No raw Flightradar24 track file is included. Historical Flightradar24 data were used upstream only to derive aggregate quantities required by the archived preprocessing, notably a transit-time percentile, origin-to-arrival-fix mapping, and aggregate 15-minute counts. The flight-level rows distributed here come from ANAC public schedule/operation records rather than from a published Flightradar24 flight table.

## BIMTRA boundary

`BIMTRA.csv` is intentionally absent. Raw BIMTRA movement records are from a DECEA internal database and are not redistributed. They are needed only if an authorized user wants to regenerate the synthetic UAM demand artifacts from the original source distribution. The committed UAM inputs are sufficient to run the published UAM experiment without BIMTRA.

## Historical preprocessing note

`data.csv` is the archived processed study input and is authoritative for rerunning the Python experiment. The historical script at `../preprocessing/R/generate_sbgr_data.R` is retained for transparency but is not guaranteed to recreate this file exactly. The current script excludes 2023-04-24 through 2023-04-28, while the archived `data.csv` contains those operational days. The repository therefore preserves both artifacts without inventing a reconciliation.

## Large files

The largest archived tables are intentionally committed directly rather than through Git LFS. They remain below GitHub's 100 MB per-file limit and are pinned by Git blob identity in `../ARTIFACTS.json`.

## Source terms

No separate data license is granted by this repository. Source datasets and derived materials remain subject to the terms of their original providers and applicable agreements.

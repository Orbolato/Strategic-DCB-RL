# Provenance and reproducibility boundaries

This repository accompanies:

> Lucas Orbolato Carvalho and Mayara Condé Rocha Murça, **“Stochastic Strategic Demand-Capacity Balancing Using Model-Free Reinforcement Learning,”** 45th Digital Avionics Systems Conference (DASC), 2026.

The repository contains the conference materials at the root and the research code/data package under `code/`. This document records what is public, what is derived, and what is intentionally excluded so that the public package does not imply broader data rights or stronger reproducibility than the archived materials support.

## Reproducibility levels

### 1. Main experiment reruns

The canonical processed inputs required by `code/maskable_ppo_test.py` and `code/maskable_ppo_test_uam.py` are committed. A fresh clone can therefore retrain and evaluate the two published case-study pipelines without access to the excluded upstream source records.

No trained policies are archived. The scripts train a model for a test day when its saved model is absent, so a first complete run is computationally expensive. The implementation fixes the documented PPO seed, and the UAM generator fixes a NumPy seed, but exact equality of newly trained neural-network weights or final numerical results is **not guaranteed** across operating systems, hardware, PyTorch builds, BLAS implementations, or library versions. The public package supports rerunning the implemented experiments; it does not claim bit-for-bit retraining reproducibility.

### 2. Public/derived input regeneration

Several intermediate inputs can be regenerated from the files committed here, including the VRA-based schedule-buffer correction, the 40-NM arrival-time approximation, the BlueSky-derived eVTOL flight-duration table, and the figures/capacity curves whose required inputs are present.

### 3. Upstream reconstruction requiring excluded source records

Some historical preprocessing paths are retained for transparency but cannot be executed from a clean clone alone. In particular:

- raw BIMTRA movement records are not redistributed;
- raw Flightradar24 track files are not redistributed;
- the earlier SBGR R/AMPL preprocessing tree references raw workspaces and local source files that are not included.

These exclusions do not prevent the two main experiment scripts from using the canonical processed inputs committed in `code/`.

## Data lineage

### ANAC SIROS and VRA

Flight-level rows distributed in this repository originate from ANAC public schedule/operation sources (SIROS and VRA) and from transformations of those public records. The committed `code/data/VRA_*.csv`, `code/data/arrival_data.csv`, and `code/arrival_time_estimation/arrival_data_with_estimated_times_40.csv` belong to this lineage.

### Flightradar24-derived information

No raw Flightradar24 trajectory file and no flight-level Flightradar24 row is distributed in this repository. Historical Flightradar24 information enters the archived preprocessing only through aggregate/derived quantities used by the study, including:

- a P10 transit-time statistic used to approximate travel time between a distance threshold and landing;
- an origin-airport-to-arrival-fix assignment derived from historical track geometry; and
- aggregate counts by 15-minute interval used in the earlier SBGR preprocessing.

The archived R scripts show how those quantities were originally produced, but the source track files themselves are excluded.

### REDEMET / DECEA weather

The SBGR processed tables contain METAR/TAF-derived weather variables. The upstream decoding scripts are retained under `code/preprocessing/`, while the experiment-ready weather tables are committed directly.

### CGNA traffic-management measures

`code/data/CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv` contains traffic-management-measure records used by the analysis. It is not a flight-level movement database.

### BIMTRA

Raw BIMTRA movement records from the DECEA internal database are intentionally excluded and ignored by `code/.gitignore`. The UAM study inputs committed as `code/data_uam.csv`, `code/arrival_data_uam.csv`, and `code/new_taf_list_uam.csv` are synthetic/derived study artifacts built from aggregate helicopter-demand patterns together with the public/committed weather and simulation-derived inputs. A clean clone can run the UAM experiment from those shipped files. Rebuilding those synthetic inputs from the original BIMTRA source requires separate authorized access to `code/data/BIMTRA.csv`.

### BlueSky simulation outputs

The state logs under `code/profile_implementation/SBGR_DEST/Logs/` are BlueSky simulation outputs generated for the eVTOL route profiles. They are not operational surveillance records. The route scenarios and the small OpenAP/UAMMETRICS additions needed to reconstruct those profiles are included for transparency.

## Relationship to the earlier runway-capacity repository

The conventional-ATM case reuses the archived SBGR processed dataset and capacity-envelope material from the earlier runway-capacity study. The committed `code/data/data.csv` has Git blob `5660ea374ddd186dff838907a8b76ec20c0f0472`, the same archived processed table preserved in the companion 2025 study package.

The retained upstream script `code/preprocessing/R/generate_sbgr_data.R` is a historical/reference preprocessing script, not a guaranteed reconstruction recipe for the archived CSV. In particular, the script currently excludes operational days 2023-04-24 through 2023-04-28, while the archived processed table contains those days. The available repository history does not establish when or why that exclusion was introduced, so both artifacts are preserved as-is and the repository does not claim exact raw-to-processed regeneration from that script.

## Large files

Two committed CSV artifacts are larger than GitHub's recommended 50 MB threshold but remain below GitHub's 100 MB hard file-size limit. They are intentionally kept in standard Git because they are canonical experiment inputs and allow a normal clone to contain the full public package without an additional large-file service.

## Licensing and source terms

No software license or data license is asserted by this repository. Public-source and third-party inputs remain subject to the terms of their original providers. Inclusion of a derived or public-source file here should not be read as relicensing the underlying source material.

## Integrity checks

`code/ARTIFACTS.json` records Git blob identities and byte sizes for the principal archived study inputs. The lightweight CI workflow verifies those anchors, checks that restricted/raw source files are absent, validates the archived SBGR dataset structure, and compiles the Python sources. The CI deliberately does **not** retrain all per-day PPO policies because that would require many hours of computation and would still not establish bit-for-bit stochastic retraining equivalence.

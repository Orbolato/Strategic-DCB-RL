# Code and data: Stochastic Strategic Demand-Capacity Balancing Using Model-Free Reinforcement Learning

Code and data for the paper:

> L. O. Carvalho and M. C. R. Murça, "Stochastic Strategic Demand-Capacity Balancing Using Model-Free Reinforcement Learning,"
> 45th Digital Avionics Systems Conference (DASC), Orlando, FL, USA, 2026.

Paper, poster and slides: [project page](https://orbolato.github.io/Strategic-DCB-RL/).

A Maskable PPO agent assigns strategic ground delays (0–4 slots of 15 min) to each arrival. A tactical heuristic then picks the runway configuration and the arrival/departure service rates under stochastic weather (METAR/TAF). The same algorithm is evaluated on two case studies:

1. **Conventional ATM at São Paulo/Guarulhos (SBGR)**: real schedules, METAR/TAF and runway configuration capacity envelopes (RCCEs).
2. **Urban Air Mobility (UAM) at SBGR**: synthetic eVTOL demand derived from aggregate helicopter-demand patterns, BlueSky flight times and non-invertible vertiport capacity curves.

For data lineage, redistribution boundaries and the distinction between experiment reruns and upstream reconstruction, see [`../PROVENANCE.md`](../PROVENANCE.md) and [`data/README.md`](data/README.md).

---

## Folder layout

```text
code/
├── maskable_ppo_test.py                       # Case 1 (SBGR conventional ATM): preprocessing, training, evaluation, figures
├── maskable_ppo_test_uam.py                   # Case 2 (UAM): preprocessing, training, evaluation, figures
├── improved_gym_sdcboptimizer_specific.py     # Gymnasium environment, conventional ATM
├── improved_gym_sdcboptimizer_specific_uam.py # Gymnasium environment, UAM
├── uam_capacity_curves.py                     # Vertiport capacity envelopes per weather condition and side (Fig. 8)
├── vertiport_feasible_region_solver.py        # Integer feasible-region enumeration used by uam_capacity_curves.py
├── generate_uam_synthetic_data_bimtra_based.py# Builds data_uam.csv, arrival_data_uam.csv, new_taf_list_uam.csv
├── analyze_bimtra.py                          # BIMTRA helicopter traffic analysis (Fig. 7; raw BIMTRA not included)
├── extract_flight_durations.py                # eVTOL block times from BlueSky logs -> flight_durations.csv
├── trajectory_plotter.py                      # eVTOL flight profile plots from a BlueSky log (Fig. 6)
├── data_uam.csv, arrival_data_uam.csv, new_taf_list_uam.csv, flight_durations.csv
├── data/
│   ├── README.md                              # Data provenance and availability notes
│   ├── data.csv                               # SBGR 15-min dataset: weather, demand, runway configuration
│   ├── new_taf_list.csv, std_taf.csv          # TAF forecasts decoded to 15-min resolution
│   ├── CGNA_MEDIDAS_ATFM_..._processed.csv    # CGNA traffic-management measures
│   ├── curves/                                # RCCEs, 3 weather x 2 configs x 8 percentiles
│   ├── arrival_data.csv                       # Flight-level SBGR arrivals from ANAC public schedule/operation lineage
│   ├── VRA_2022.csv, VRA_2023.csv, VRA_2024.csv
│   └── (BIMTRA.csv)                           # Restricted raw BIMTRA records; intentionally not included
├── arrival_time_estimation/
│   ├── estimate_real_arrivals.py              # VRA-based schedule-buffer correction
│   ├── compute_arrival_40_times.py            # 40-NM timing approximation and 15-min slot assignment
│   └── arrival_data_with_estimated_times_40.csv
├── profile_implementation/
│   ├── routes_to_scn.py, routes_uam_detailed_rmsp.csv
│   ├── SBGR_DEST/*.scn
│   ├── SBGR_DEST/Logs/STATELOG_*.log          # BlueSky simulation outputs, not surveillance data
│   └── bluesky/
├── preprocessing/                             # Historical upstream R/AMPL code; reference only
├── ARTIFACTS.json                             # Git-object anchors for principal archived inputs
└── tests/test_repository_integrity.py         # Lightweight public-package integrity checks
```

## Installation

Tested with Python 3.12.3 on Windows.

```bash
git clone https://github.com/Orbolato/Strategic-DCB-RL.git
cd Strategic-DCB-RL/code
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Windows:

- Clone to a short path. Some nested files can exceed the legacy 260-character path limit when the parent folder is deep.
- Set `PYTHONUTF8=1` if redirected console output encounters Unicode check marks.

## Reproducing the main experiments

Run the scripts **from this `code/` folder**. Paths in the historical scripts are relative to it.

```bash
python maskable_ppo_test.py       # Case 1: SBGR conventional ATM (Tables I-II, Figs. 3, 9-12)
python maskable_ppo_test_uam.py   # Case 2: UAM (Tables III-IV, Figs. 13-16)
```

For each of the 30 test days, the scripts train a Maskable PPO model and save it, with result figures, to `Results/` (Case 1) or `Results_UAM/` (Case 2). A model already present there is loaded instead of retrained. No trained policy files are committed to this repository, so a fresh clone performs training on the first full run.

Training uses 500,000 timesteps per day for Case 1 and 300,000 per day for Case 2. In the original environment, this was approximately 1 hour per day for Case 1 and 40 minutes per day for Case 2 on one Intel i7-9700KF core. The days are independent and can be trained separately. These timings are historical measurements, not performance guarantees.

Each day is evaluated three ways:

- **Baseline**: tactical heuristic with no strategic delay (`g_f = 0`) under observed weather (METAR).
- **Deploy**: policy selects delays deterministically using forecast weather (TAF).
- **Test**: those delays are replayed under observed weather (METAR), giving the reported tactical cost.

The console prints the statistics used in Tables I/III (`[1] DESCRIPTIVE STATISTICS`), the delay distribution used in Tables II/IV (`ALLOCATED DELAYS DISTRIBUTION ANALYSIS`), and the Wilcoxon signed-rank test. A final section also performs a stochastic-weather evaluation on five days that is not reported in the paper.

### Reproducibility boundary

The committed CSVs are the canonical experiment inputs. The implementation fixes the documented PPO seed, but exact equality of newly trained neural-network weights or final numerical outputs is **not guaranteed** across operating systems, hardware, PyTorch builds or numerical libraries. CI therefore verifies the archived inputs, repository contracts and Python syntax; it does not attempt all-day PPO retraining.

### Main settings (as implemented)

| Item | Value |
|---|---|
| Test days (both cases) | 2022-2-16, 2022-3-25, 2022-5-26, 2022-5-27, 2022-5-28, 2022-6-9, 2022-6-12, 2022-6-13, 2022-8-17, 2022-8-24, 2022-10-23, 2022-11-7, 2022-11-27, 2022-12-2, 2023-2-9, 2023-2-15, 2023-2-28, 2023-3-25, 2023-5-8, 2023-5-19, 2023-8-6, 2023-8-16, 2023-9-25, 2023-10-20, 2023-10-22, 2023-10-27, 2023-11-1, 2023-11-15, 2023-11-30, 2023-12-12 |
| Planning horizon | 96 slots of 15 min (operational day 06:00 to 05:45 next day) |
| Action | `g_f ∈ {0, …, min(4, T − t)}`, enforced by action masks |
| Cost weights | α = 2 (airborne vs ground), β = 1, δ = 0.3 (configuration switch), η = 0.1 (strategic delay) |
| Wind limits | crosswind ≤ 20 kt, tailwind ≤ 5 kt; for UAM, applied to the two vertiport sides |
| Weather probabilities (Eqs. 14, 16) | METAR transitions between consecutive slots and TAF/METAR pairs from all days except the test days |
| RCCE percentile (Case 1) | VMC 0.99, IMC 0.95, LIMC 0.90 |
| PPO | `sb3_contrib.MaskablePPO("MlpPolicy")` with γ = 1, learning rate 3e-4, batch 64, entropy coefficient 0.01, clip 0.2, seed 0 (it also seeds the environment's weather sampler, Gymnasium's `np_random`, at the first reset), trained on the CPU with one thread so that reruns with the same seed and software stack give identical policies; other settings follow sb3-contrib 2.7.1 defaults used by the archived environment |

## Figures and tables

| Paper | Script | Output |
|---|---|---|
| Fig. 3 (RCCEs) | `maskable_ppo_test.py` | `rcces_gru.pdf` |
| Fig. 4 (vertiport candidates and routes) | from ref. [31] | route waypoints in `profile_implementation/routes_uam_detailed_rmsp.csv` |
| Fig. 5 (BlueSky screenshot) | BlueSky GUI with `profile_implementation/SBGR_DEST/SBSP-SBGR.scn` | — |
| Fig. 6 (altitude vs distance) | `trajectory_plotter.py` | `trajectory_plots/altitude_vs_distance.pdf` |
| Fig. 7 (helicopter departures by hour) | `analyze_bimtra.py` | `output/helicopter_analysis_bimtra/04_hourly_operations.pdf` |
| Fig. 8 (UAM capacity curves) | `uam_capacity_curves.py` | `uam_capacity_curves_all_combined.pdf` |
| Fig. 9 / 13 | `maskable_ppo_test.py` / `maskable_ppo_test_uam.py` | `Results[_UAM]/sdcb_vs_baseline_violin.pdf` |
| Fig. 10 / 14 | same | `Results[_UAM]/sdcb_improvements_sorted_all_nofilter.pdf` |
| Fig. 11 / 15 | same | `Results[_UAM]/sdcb_baseline_vs_pct_improvement.pdf` |
| Fig. 12 / 16 | same | `Results[_UAM]/sdcb_pct_improvements_sorted.pdf` |
| Tables I–IV | same | console output |

Figs. 1 and 2 are schematic diagrams. The paper uses the UAM figures under a `uam_` prefix.

## Regenerating inputs (optional)

The repository includes every input required by the two main experiment scripts. The steps below describe optional upstream regeneration. Some source records needed to rebuild every processed input from first principles are intentionally excluded.

### 1. SBGR arrival demand (buffer-corrected, 40 NM)

```bash
python arrival_time_estimation/estimate_real_arrivals.py
python arrival_time_estimation/compute_arrival_40_times.py
```

The flight-level rows come from the ANAC public schedule/operation lineage. The 40-NM approximation uses a historical aggregate transit-time statistic; no per-flight Flightradar24 track row is distributed in the public package.

### 2. eVTOL flight times (BlueSky)

`profile_implementation/routes_to_scn.py` writes scenarios from `routes_uam_detailed_rmsp.csv`. Running the archived scenarios requires [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) plus the two small additions in `profile_implementation/bluesky/`:

- copy `uam_metrics.py` to `bluesky/plugins/`;
- merge the `EVTOL` entry of `evtol_openap_rotor.json` into `bluesky/resources/performance/OpenAP/rotor/aircraft.json`.

Then:

```bash
python extract_flight_durations.py
python trajectory_plotter.py
```

The committed `STATELOG_*.log` files are BlueSky simulation outputs, not operational surveillance records.

### 3. Synthetic UAM demand

Rebuilding the synthetic UAM inputs from the original helicopter-demand distribution requires an authorized local `data/BIMTRA.csv`, which is intentionally excluded:

```bash
python analyze_bimtra.py
python generate_uam_synthetic_data_bimtra_based.py
```

The generator samples daily totals, 15-minute slots and origins pseudo-randomly using the fixed `RANDOM_SEED` in the script. The committed UAM CSVs preserve the experiment draw and remain the canonical inputs. With the same source data, code path and NumPy behavior, the fixed seed is intended to reproduce the same draw; exact upstream reconstruction is not claimed across arbitrary environments.

### 4. Upstream SBGR dataset and RCCEs

`preprocessing/` holds historical R and AMPL code from the earlier study (Carvalho and Murça, TR-A 2025) that assembled the processed SBGR tables and RCCEs. It is retained for transparency and does not run as-is from a clean clone because the original raw workspaces are not distributed. See [`preprocessing/README.md`](preprocessing/README.md).

An additional provenance caveat applies: the archived `data/data.csv` is authoritative for the Python experiment, while the current historical `generate_sbgr_data.R` excludes 2023-04-24 through 2023-04-28 even though the archived CSV contains those days. The repository preserves both artifacts without asserting an undocumented reconciliation.

## Data availability

| File(s) | Source | Notes |
|---|---|---|
| `data/data.csv`, `data/new_taf_list.csv`, `data/std_taf.csv` | REDEMET/DECEA weather; ANAC SIROS/VRA traffic; aggregate historical Flightradar24-derived quantities used upstream | Processed 15-min tables, 2022-01-02 to 2023-12-30. No raw or flight-level Flightradar24 track file is distributed. |
| `data/arrival_data.csv`, `arrival_time_estimation/arrival_data_with_estimated_times_40.csv` | ANAC SIROS/VRA flight-level rows; 40-NM timing uses an aggregate historical transit-time statistic | 235,075 arrivals. No per-flight Flightradar24 row is redistributed. |
| `data/VRA_*.csv` | ANAC VRA public data | Used for historical flight-time deviation estimates. |
| `data/CGNA_MEDIDAS_ATFM_*.csv` | CGNA ATFM measures | Traffic-management-measure records used by the conventional-case analysis; not a flight movement database. |
| `data/BIMTRA.csv` | BIMTRA (DECEA internal database) | **Not included.** Required only for authorized regeneration of the synthetic UAM demand from the original source distribution. |
| `data_uam.csv`, `arrival_data_uam.csv`, `new_taf_list_uam.csv` | Synthetic study artifacts | Experiment-ready UAM inputs derived from committed weather/simulation inputs and aggregate helicopter-demand patterns. |

No software or data license is asserted by this repository. Source terms remain with the original providers.

## Citation

Please cite the paper (metadata are also in [`../CITATION.cff`](../CITATION.cff)):

```bibtex
@inproceedings{carvalho2026stochastic,
  author    = {Carvalho, Lucas Orbolato and Mur{\c{c}}a, Mayara Cond{\'e} Rocha},
  title     = {Stochastic Strategic Demand-Capacity Balancing Using Model-Free Reinforcement Learning},
  booktitle = {45th Digital Avionics Systems Conference (DASC)},
  address   = {Orlando, FL, USA},
  year      = {2026}
}
```

## Acknowledgments

Supported by the Brazilian Civil Aviation Secretariat (SAC), Decentralized Execution Agreement (TED) No. 11525720240005-003882/2024, and by the Brazilian Air Force (FAB).

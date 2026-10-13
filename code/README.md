# Code and data: Stochastic Strategic Demand-Capacity Balancing Using Model-Free Reinforcement Learning

Code and data for the paper:

> L. O. Carvalho and M. C. R. Murça, "Stochastic Strategic Demand-Capacity Balancing Using Model-Free Reinforcement Learning,"
> 45th Digital Avionics Systems Conference (DASC), Orlando, FL, USA, 2026.

Paper, poster and slides: [project page](https://orbolato.github.io/Strategic-DCB-RL/).

A Maskable PPO agent assigns strategic ground delays (0–4 slots of 15 min) to each arrival. A tactical heuristic
then picks the runway configuration and the arrival/departure service rates under stochastic weather
(METAR/TAF). The same algorithm is evaluated on two case studies:

1. **Conventional ATM at São Paulo/Guarulhos (SBGR)**: real schedules, METAR/TAF and runway configuration capacity envelopes (RCCEs).
2. **Urban Air Mobility (UAM) at SBGR**: synthetic eVTOL demand built from BIMTRA helicopter patterns, BlueSky flight times and non-invertible vertiport capacity curves.

---

## Folder layout

```
code/
├── maskable_ppo_test.py                       # Case 1 (SBGR conventional ATM): preprocessing, training, evaluation, figures
├── maskable_ppo_test_uam.py                   # Case 2 (UAM): preprocessing, training, evaluation, figures
├── improved_gym_sdcboptimizer_specific.py     # Gymnasium environment, conventional ATM (RunwayEnv)
├── improved_gym_sdcboptimizer_specific_uam.py # Gymnasium environment, UAM (RunwayEnv with v_min/v_max curves)
├── uam_capacity_curves.py                     # Vertiport capacity envelopes per weather condition and side (Fig. 8)
├── vertiport_feasible_region_solver.py        # Integer feasible-region enumeration used by uam_capacity_curves.py
├── generate_uam_synthetic_data_bimtra_based.py# Builds data_uam.csv, arrival_data_uam.csv, new_taf_list_uam.csv
├── analyze_bimtra.py                          # BIMTRA helicopter traffic analysis (Fig. 7)
├── extract_flight_durations.py                # eVTOL block times from BlueSky logs -> flight_durations.csv
├── trajectory_plotter.py                      # eVTOL flight profile plots from a BlueSky log (Fig. 6)
├── data_uam.csv, arrival_data_uam.csv, new_taf_list_uam.csv, flight_durations.csv   # UAM inputs
├── data/
│   ├── data.csv                               # SBGR 15-min dataset: METAR weather, wind components, demand, runway config
│   ├── new_taf_list.csv, std_taf.csv          # TAF forecasts decoded to 15-min resolution
│   ├── CGNA_MEDIDAS_ATFM_..._processed.csv    # CGNA ATFM measures (MIT/GDP), descriptive context only
│   ├── curves/                                # RCCEs (quantile-regression capacity envelopes), 3 weather x 2 configs x 8 percentiles
│   ├── arrival_data.csv                       # Flight-level SBGR arrivals (input to arrival_time_estimation/)
│   ├── VRA_2022.csv, VRA_2023.csv, VRA_2024.csv   # ANAC VRA observed flight times
│   └── (BIMTRA.csv)                           # Raw BIMTRA records, not included (see Data availability)
├── arrival_time_estimation/
│   ├── estimate_real_arrivals.py              # Removes schedule buffers using VRA flight-time deviations (P10 per origin+equipment)
│   ├── compute_arrival_40_times.py            # 40 NM times and 15-min slot assignment
│   └── arrival_data_with_estimated_times_40.csv   # Flight-level demand used by Case 1
├── profile_implementation/
│   ├── routes_to_scn.py, routes_uam_detailed_rmsp.csv   # Route waypoints -> BlueSky scenarios (REH-based routes)
│   ├── SBGR_DEST/*.scn                        # The five scenarios simulated (VP-001/2/3, SBSP, SBMT -> SBGR)
│   ├── SBGR_DEST/Logs/STATELOG_*.log          # BlueSky state logs used for flight durations and Fig. 6
│   └── bluesky/                               # BlueSky add-ons: UAMMETRICS plugin and EVTOL performance entry
└── preprocessing/                             # Upstream R/AMPL code that built data/ (reference only, see below)
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

On Windows, two things to watch:
- Clone to a short path. Some nested files exceed the 260-character path limit when the parent folder is deep.
- Set `PYTHONUTF8=1` if you redirect script output to a file, because the scripts print Unicode check marks.

## Reproducing the results

Run the scripts **from this `code/` folder**. All paths are relative to it.

```bash
python maskable_ppo_test.py       # Case 1: SBGR conventional ATM (Tables I-II, Figs. 3, 9-12)
python maskable_ppo_test_uam.py   # Case 2: UAM (Tables III-IV, Figs. 13-16)
```

For each of the 30 test days, the scripts train a Maskable PPO model and save it, with the result figures, to
`Results/` (Case 1) or `Results_UAM/` (Case 2). A model that already exists there is loaded instead of retrained,
so a second run only evaluates. Training takes one model per day, 500,000 timesteps (Case 1) or 300,000 timesteps
(Case 2), where one timestep is one flight's ground-delay decision. On one core of an Intel i7-9700KF this is about
1 hour per day for Case 1 and 40 minutes per day for Case 2. The days are independent, so they can be trained in
parallel on separate cores; a GPU gives little benefit because the environment simulation dominates the run time.

Each day is evaluated three ways:

- **Baseline**: the tactical heuristic with no strategic delay (g_f = 0), on observed weather (METAR).
- **Deploy**: the policy picks delays deterministically on forecast weather (TAF).
- **Test**: those delays are replayed on observed weather (METAR), giving the reported tactical cost.

The console prints the statistics in Tables I/III (section `[1] DESCRIPTIVE STATISTICS`), the delay distribution
in Tables II/IV (`ALLOCATED DELAYS DISTRIBUTION ANALYSIS`) and the Wilcoxon signed-rank test. A final section
also runs a stochastic-weather evaluation on 5 days, which is not reported in the paper.

### Main settings (as implemented)

| Item | Value |
|---|---|
| Test days (both cases) | 2022-2-16, 2022-3-25, 2022-5-26, 2022-5-27, 2022-5-28, 2022-6-9, 2022-6-12, 2022-6-13, 2022-8-17, 2022-8-24, 2022-10-23, 2022-11-7, 2022-11-27, 2022-12-2, 2023-2-9, 2023-2-15, 2023-2-28, 2023-3-25, 2023-5-8, 2023-5-19, 2023-8-6, 2023-8-16, 2023-9-25, 2023-10-20, 2023-10-22, 2023-10-27, 2023-11-1, 2023-11-15, 2023-11-30, 2023-12-12 |
| Planning horizon | 96 slots of 15 min (operational day 06:00 to 05:45 next day, data timestamps) |
| Action | g_f ∈ {0, …, min(4, T − t)}, enforced by action masks |
| Cost weights | α = 2 (airborne vs ground), β = 1, δ = 0.3 (configuration switch), η = 0.1 (strategic delay) |
| Wind limits | crosswind ≤ 20 kt, tailwind ≤ 5 kt; for UAM, applied to the two vertiport sides, which are perpendicular to the runways |
| Weather probabilities (Eqs. 14, 16) | METAR transitions between consecutive slots and TAF/METAR pairs from all days except the test days |
| RCCE percentile (Case 1) | VMC 0.99, IMC 0.95, LIMC 0.90 (from `data/curves/`) |
| PPO | `sb3_contrib.MaskablePPO("MlpPolicy")` with γ = 1, learning rate 3e-4, batch 64, entropy coefficient 0.01, clip 0.2, seed 0 (`PPO_KWARGS`); other settings are sb3-contrib 2.7.1 defaults (n_steps 2048, 10 epochs) |

## Figures and tables

| Paper | Script | Output |
|---|---|---|
| Fig. 3 (RCCEs) | `maskable_ppo_test.py` | `rcces_gru.pdf` |
| Fig. 4 (vertiport candidates and routes) | from ref. [31] | route waypoints in `profile_implementation/routes_uam_detailed_rmsp.csv` |
| Fig. 5 (BlueSky screenshot) | BlueSky GUI with `profile_implementation/SBGR_DEST/SBSP-SBGR.scn` | — |
| Fig. 6 (altitude vs distance) | `trajectory_plotter.py` | `trajectory_plots/altitude_vs_distance.pdf` |
| Fig. 7 (helicopter departures by hour) | `analyze_bimtra.py` | `output/helicopter_analysis_bimtra/04_hourly_operations.pdf` |
| Fig. 8 (UAM capacity curves) | `python uam_capacity_curves.py` | `uam_capacity_curves_all_combined.pdf` |
| Fig. 9 / 13 | `maskable_ppo_test.py` / `maskable_ppo_test_uam.py` | `Results[_UAM]/sdcb_vs_baseline_violin.pdf` |
| Fig. 10 / 14 | same | `Results[_UAM]/sdcb_improvements_sorted_all_nofilter.pdf` |
| Fig. 11 / 15 | same | `Results[_UAM]/sdcb_baseline_vs_pct_improvement.pdf` |
| Fig. 12 / 16 | same | `Results[_UAM]/sdcb_pct_improvements_sorted.pdf` |
| Tables I–IV | same | console output |

Figs. 1 and 2 are schematic diagrams. The paper uses the UAM figures under a `uam_` prefix.

## Regenerating the inputs (optional)

This folder includes every input the experiments need. The steps below rebuild them.

**1. SBGR arrival demand (buffer-corrected, 40 NM)**
```bash
python arrival_time_estimation/estimate_real_arrivals.py     # data/arrival_data.csv + data/VRA_*.csv
python arrival_time_estimation/compute_arrival_40_times.py   # -> arrival_time_estimation/arrival_data_with_estimated_times_40.csv
```

**2. eVTOL flight times (BlueSky)**
- `profile_implementation/routes_to_scn.py` writes scenarios from `routes_uam_detailed_rmsp.csv` (climb to 300 ft and cruise at 1,500 ft AGL, 48/96 kt).
  The five files in `SBGR_DEST/` are those scenarios, reversed to end at SBGR, with the initial heading set to 0.
- Running them needs [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) with two additions from `profile_implementation/bluesky/`:
  - copy `uam_metrics.py` to `bluesky/plugins/`;
  - merge the `EVTOL` entry of `evtol_openap_rotor.json` into `bluesky/resources/performance/OpenAP/rotor/aircraft.json`.
- Then:
```bash
python extract_flight_durations.py   # SBGR_DEST/Logs -> flight_durations.csv
python trajectory_plotter.py         # Fig. 6
```

**3. Synthetic UAM demand** (needs `data/BIMTRA.csv`)
```bash
python analyze_bimtra.py                              # Fig. 7 and output/helicopter_analysis_bimtra/BIMTRA_departures.csv
python generate_uam_synthetic_data_bimtra_based.py    # -> data_uam.csv, arrival_data_uam.csv, new_taf_list_uam.csv
```
The generator draws daily totals, slots and flight origins at random (`RANDOM_SEED` in the script). The shipped
files hold the demand draw used in the experiments; a new run produces a different draw.

**4. Upstream SBGR dataset and RCCEs**: `preprocessing/` holds the R and AMPL code from the earlier study
(Carvalho and Murça, TR-A 2025) that assembled `data/data.csv`, decoded the TAFs and fitted the RCCEs
(Eqs. 7–8, `linear_quantreg.mod`). It is included for transparency and does not run as-is; see
`preprocessing/README.md`.

## Data availability

| File(s) | Source | Notes |
|---|---|---|
| `data/data.csv`, `data/new_taf_list.csv`, `data/std_taf.csv` | METAR/TAF from REDEMET (DECEA); demand from ANAC SIROS/VRA and Flightradar24 tracks | Processed 15-min aggregates, 2022-01-02 to 2023-12-30 |
| `data/arrival_data.csv`, `arrival_time_estimation/arrival_data_with_estimated_times_40.csv` | ANAC SIROS/VRA; 40/100 NM times from Flightradar24 tracks | Flight level, 235,075 arrivals |
| `data/VRA_*.csv` | ANAC VRA (open data) | |
| `data/CGNA_MEDIDAS_ATFM_*.csv` | CGNA ATFM measures | Used only for descriptive plots |
| `data/BIMTRA.csv` | BIMTRA (DECEA internal database) | Raw records with aircraft registrations. **Not included.** The UAM experiments run from the shipped UAM CSVs without it; it is only needed to regenerate them (step 3) |
| `data_uam.csv`, `arrival_data_uam.csv`, `new_taf_list_uam.csv` | Synthetic (this work) | SBGR METAR/TAF + BIMTRA-based demand |

## Citation

Please cite the paper (metadata also in [`CITATION.cff`](../CITATION.cff)):

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

Supported by the Brazilian Civil Aviation Secretariat (SAC), Decentralized Execution Agreement (TED)
No. 11525720240005-003882/2024, and by the Brazilian Air Force (FAB).

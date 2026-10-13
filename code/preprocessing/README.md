# Upstream preprocessing (reference only)

This folder holds the R and AMPL code from the earlier study (L. O. Carvalho and M. C. R. Murça, "A stochastic
model-free reinforcement learning framework for optimizing runway capacity management under uncertainty,"
*Transp. Res. Part A*, vol. 200, 2025) that produced the processed SBGR inputs in `../data/`.

It is included to document how those inputs were built. **It does not run as-is.** The scripts
use absolute `setwd()` paths from the original workspace, and they need raw inputs that are not distributed here:
VRA/SIROS dumps, Flightradar24-derived track files, METAR/TAF archives, and intermediate `.RData` workspaces.

| File | Role | Output used here |
|---|---|---|
| `R/taf_decoder.R`, `R/forecast_data_decoder_mod.R` | Decode raw TAF messages into 15-min weather condition and wind state | `data/new_taf_list.csv`, `data/std_taf.csv` |
| `R/generate_sbgr_data.R` | Merge METAR, schedules (VRA/SIROS) and 40/100 NM times into the 15-min SBGR dataset | `data/data.csv` (the final `write.csv` call is commented out in this version) |
| `R/generate_flight_data.R` | Build the flight-level arrival table | `data/arrival_data.csv` |
| `R/capacity_management.R` + `ampl/linear_quantreg.mod` | Fit runway configuration capacity envelopes by piecewise-linear quantile regression (paper Eqs. 7–8), solved with AMPL/Gurobi | `data/curves/*_RCCE_<p>.csv` |

RCCE files are named `<WC>_<CONFIG>_RCCE_<p>.csv`, where `<p>` is the percentile without the leading `0.` (e.g.
`9900` = 0.99). `maskable_ppo_test.py` uses 0.99 for VMC, 0.95 for IMC and 0.90 for LIMC.

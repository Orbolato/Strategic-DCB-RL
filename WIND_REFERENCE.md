# Wind-direction reference convention

For the **conventional ATM/SBGR case study** in this repository, runway-relative wind information follows a **true-north reference frame**.

The conventional ATM case reuses the SBGR processed weather/capacity inputs and historical preprocessing lineage of the earlier runway-capacity studies. METAR and TAF wind directions are meteorological directions referenced to true north, and the historical preprocessing compares them with the SBGR runway axis expressed in the same frame:

- runway 10 direction: **73° true**;
- reciprocal runway 28 direction: **253° true**.

The 73°/253° values are rounded true runway-axis bearings. They are **not** the runway designator values interpreted as 100°/280°, and they are not magnetic headings. Thus the runway/wind compatibility calculation uses a consistent true-north reference.

The relevant historical preprocessing is retained under `code/preprocessing/R/`, including `generate_sbgr_data.R` and `forecast_data_decoder_mod.R`. The conventional ATM reinforcement-learning experiment consumes the processed wind state from the shipped SBGR inputs rather than applying a second magnetic/true conversion.

This note is scoped to the conventional ATM/SBGR runway-capacity case. The repository also contains a separate UAM case study with its own network/capacity modeling and should not be interpreted as using the SBGR runway-axis convention for all wind-related quantities.

A later SBGR runway-configuration forecasting study uses more precise true bearings of approximately 73.69°/253.67°. This repository intentionally preserves the rounded 73°/253° convention associated with its SBGR preprocessing lineage. This documentation does not modify code, data, policies, or reported results.

# Stochastic Strategic Demand-Capacity Balancing Using Model-Free Reinforcement Learning

Public paper page, conference materials, and code and data for:

**Lucas Orbolato Carvalho and Mayara Condé Rocha Murça, “Stochastic Strategic Demand-Capacity Balancing Using Model-Free Reinforcement Learning,” 45th Digital Avionics Systems Conference (DASC), 2026.**

The work was selected as a **DASC 2026 Student Research Competition finalist** (Paper ID 276).

## Public materials

- Project page: https://orbolato.github.io/Strategic-DCB-RL/
- Paper PDF: https://orbolato.github.io/Strategic-DCB-RL/paper.pdf
- Poster: https://orbolato.github.io/Strategic-DCB-RL/poster.pdf
- Presentation slides: https://orbolato.github.io/Strategic-DCB-RL/slides.pdf
- Code and data: [`code/`](code/) (instructions in [`code/README.md`](code/README.md))
- Provenance and reproducibility boundaries: [`PROVENANCE.md`](PROVENANCE.md)
- Author publication hub: https://orbolato.github.io/publications/

## Repository scope

This repository has two parts:

- **Project page** (repository root): the public paper, poster, presentation slides, and scholarly metadata used by the website. It is deployed to GitHub Pages by `.github/workflows/deploy-pages.yml`.
- **Code and data** ([`code/`](code/)): the reinforcement-learning environments, experiment scripts, canonical processed inputs, and data-preparation pipelines for both case studies (SBGR conventional operations and UAM). The `code/` folder is intentionally excluded from the deployed website artifact.

## Reproducibility scope

The public package contains the canonical inputs needed by the two main experiment scripts, but it does **not** contain archived trained policies. The scripts therefore retrain policies when no saved model is present. A fixed PPO seed is documented in the code, but exact equality of newly trained neural-network weights or reported numerical outputs is not guaranteed across operating systems, hardware, or library builds.

**Implementation note.** The experiment implementation used the sb3-contrib default entropy coefficient of 0 and linear queue/configuration-switch penalties during PPO training. The quadratic queue-based formulation was used for evaluation and reported operational costs. These implementation details differ from the corresponding PPO/reward specifications reported in the conference paper; the repository preserves the settings actually used to generate the experimental results.

Some upstream reconstruction paths require source material that is intentionally not redistributed. In particular, raw BIMTRA movement records and raw Flightradar24 track files are excluded. The shipped experiment inputs remain usable without those restricted/external raw records. See [`PROVENANCE.md`](PROVENANCE.md) and [`code/data/README.md`](code/data/README.md) for the detailed boundary.

## Citation

Machine-readable citation metadata are provided in `CITATION.cff`. When referring to the research results or using the code and data, cite the conference paper rather than the repository.

## License and source terms

No software or data license is asserted by this repository. Source datasets remain subject to the terms of their original providers.

## Related work

For the broader research portfolio, including journal articles and companion code repositories, see:

https://orbolato.github.io/publications/

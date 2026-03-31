# CMPE492 – Senior Project

## Project Description
This repository contains the senior project work for the CMPE492 course.
The project topic will be finalized in consultation with the project advisor.

## Objectives
- Identify and define a suitable senior project topic
- Design and implement a technically sound solution
- Apply software engineering best practices
- Document the entire development process

## Technologies
- Programming Language(s): To be determined
- Frameworks & Tools: To be determined
- Version Control: GitHub

## Project Structure
- **`rd/`** — Gray–Scott reaction–diffusion experiments: target generation, parameter sweeps, genetic algorithm (`F`,`k`), fitness ablations, figure scripts.
- **`outputs/`** — Generated assets (tracked where useful for the report):
  - `outputs/target/` — reference `target_pattern.npy` / `.png`
  - `outputs/ga/` — GA best snapshots and fitness curves (`best_gen/`, `best_radial_gen/`, `best_components_gen/`)
  - `outputs/sweeps/` — `sweep_outputs/`, `refined_sweep_outputs/` (patterns, CSV, heatmaps)
  - `outputs/visual_checks/` — FFT / component / radial / comparison figures
- **`scripts/`** — ML pipeline: dataset generation, CNN training, evaluation (`scripts/models/` for weights and eval PNGs).
- **`data/processed/`** — Large `rd_dataset.npz` (gitignored); create with `python scripts/generate_ml_dataset.py` from repo root.

Run Python scripts from the **repository root** so paths resolve correctly.

## Team Member
- **Ceyda İrem Irwin** – 2023400342

## Advisor
- Arzucan Özgür Türkmen

## Project Timeline
- **Project Plan (Wiki):** Week 3  
- **Midterm Report:** 31 March 2026  
- **Final Report & Poster Presentation:** 11 June 2026  

## Notes
This repository will be updated regularly in accordance with the CMPE492 project requirements.
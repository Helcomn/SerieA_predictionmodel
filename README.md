# Football Match Prediction and Value Betting Meta-Model

Machine learning system for predicting football match outcomes across the top 5 European leagues: England, Spain, Italy, Germany, and France. The pipeline combines Elo ratings, a Poisson goal model with Dixon-Coles adjustment, market odds, and stacked meta-models to estimate match probabilities and evaluate value betting strategies.

Developed as part of a diploma thesis in Computer and Informatics Engineering.

## Features

- Historical and fixture data sync from external CSV sources
- Dynamic Elo ratings with home advantage, goal-margin updates, and promoted-team initialization
- Recency-weighted Poisson team strengths with Dixon-Coles low-score correction
- Meta-features built from model probabilities, market probabilities, Elo differences, expected goals, and momentum
- XGBoost meta-model plus MLP baseline
- Probability calibration with temperature scaling
- Ensemble blending across base model, market, XGBoost, and MLP outputs
- Fixed temporal train / validation / test split with per-league chronological processing
- Evaluation with log loss, Brier score, ECE, accuracy, betting simulation, and audit reports
- Upcoming matchday predictions from dedicated fixture files

## Repository Structure

- `main.py`: training and evaluation pipeline entry point
- `predict_match.py`: interactive custom match predictor entry point
- `src/trainer.py`: end-to-end pipeline orchestration
- `src/predictor.py`: runtime predictor utilities
- `src/services/upcoming.py`: upcoming matchday predictions

## Installation

Requires Python 3.9+.

### Windows

1. Clone the repository.
2. Run:

```powershell
.\setup.ps1
```

3. Activate the environment:

```powershell
.\venv\Scripts\activate
```

### Linux / macOS

1. Clone the repository.
2. Make the setup script executable.
3. Run:

```bash
chmod +x setup.sh
./setup.sh
```

4. Activate the environment:

```bash
source venv/bin/activate
```

## Usage

### 1. Update Data

Download the latest historical results and future fixtures:

```bash
python src/update_data.py
```

### 2. Run the Main Pipeline

Runs tuning or artifact loading, model fitting or loading, evaluation, betting simulation, and upcoming matchday picks:

```bash
python main.py
```

### 3. Predict a Custom Match

Run the root entry point from the project root:

```bash
python predict_match.py
```

## Methodology Overview

```text
Raw match data
  -> Elo ratings
  -> Recency-weighted Poisson team strengths
  -> Dixon-Coles outcome probabilities
  -> Market implied probabilities
  -> Auxiliary features: Elo diff, xG, momentum
  -> Meta-features
  -> XGBoost / MLP / blended ensemble
  -> Final probabilities
  -> Betting simulation and upcoming picks
```

## Evaluation

The main reported metrics are:

- Log loss
- Multiclass Brier score
- Expected calibration error
- Accuracy

The training pipeline uses:

- a fixed date-based split defined in `src/config.py`
- league-level parameter tuning on the validation period
- early / late validation sub-splits for meta-model and blend tuning

Betting evaluation is a simulation, not a claim of deployable profitability. The current implementation uses:

- edge threshold `0.05`
- fractional Kelly staking with `kelly_fraction=0.25`
- max stake cap `0.05`
- filters for extreme odds and extreme theoretical EV

Reported betting outputs include bet count, hit rate, profit, ROI, average odds, and additional audit tables.

## Data Sources

- Historical match results and bookmaker odds: `football-data.co.uk`
- Future fixtures: `fixturedownload.com`

## Notes

- Upcoming fixtures come from dedicated fixture CSV files.
- Upcoming picks are limited to the current or next available matchday window.
- Hyperparameter tuning uses Optuna when available; otherwise the active code falls back to a smaller manual search.
- The README describes the active refactored path, not older duplicate modules that may have existed during development.

## Thesis Context

This repository evolved from an earlier narrower prototype into a modular top-5-leagues project. The current active code path is the refactored one described above.

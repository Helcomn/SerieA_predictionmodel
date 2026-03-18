# Football Match Prediction & Value Betting Meta-Model

A machine learning system for predicting football match outcomes across the top 5 European leagues (England, Spain, Italy, Germany, France). The system combines classical statistical models (Elo Ratings, Poisson Distribution with Dixon-Coles correction) with modern ensemble learning and a deep-learning baseline to estimate true match probabilities and identify value bets against bookmaker odds.

> *Developed as part of a Diploma Thesis in Computer & Informatics Engineering.*

---

## Features

* **Automated Data Pipeline:** Downloads and syncs historical match data and upcoming fixture files automatically.
* **Elo Rating System:** Dynamic team strength ratings updated after every match, with configurable home advantage, goal-margin multipliers, and dynamic rating initialization for newly promoted teams.
* **Form / Momentum Tracking:** Calculates short-term performance trends (momentum) over a sliding window of recent matches to capture team form.
* **Poisson Goal Model:** Per-team attack/defense strength estimation with exponential time-decay weighting and Dixon-Coles low-score correction.
* **Bayesian Hyperparameter Optimization:** Automated tuning of XGBoost and MLP models using Optuna for efficient exploration of the parameter space.
* **XGBoost Meta-Model:** Learns non-linear patterns from Poisson+Elo probabilities, market odds, and auxiliary features (e.g., Elo diff, xG, momentum).
* **MLP Deep Learning Baseline:** A neural-network classifier trained on the same meta-feature space and calibrated separately.
* **Ensemble Blender:** Final weighted ensemble combining Base Model, Market, XGBoost and MLP outputs.
* **Temperature Scaling Calibration:** Post-hoc probability calibration using logit-space temperature scaling.
* **Walk-Forward Backtesting:** Strictly temporal train/validation/test splits with no data leakage.
* **Value Betting Simulation:** ROI, Net Profit and Hit Rate breakdown by market segment.
* **Per-League Metrics:** LogLoss, Brier Score and ECE reported individually for each league after evaluation.
* **Upcoming Matchday Picks:** Predicts only the current / next available league matchday from the fixture files while preventing leakage from already-played matches.

## Installation
Requires Python 3.9+.
## Windows
### 1. Clone the repository
```cmd
git clone https://github.com/Dimprassos/SerieA_predictionmodel.git
```
### 2. Run the setup script
```powershell
.\setup.ps1
```
### 3. Activate the virtual environment
```cmd
venv\Scripts\activate
```
---
## Linux / macOS / other Unix-like environments
## 1. Clone the repository
```bash
git clone https://github.com/Dimprassos/SerieA_predictionmodel.git
```
## 2. Make the setup script executable
```bash
chmod +x setup.sh
```
## 3. Run the setup script
```bash
./setup.sh
```
## 4. Activate the virtual environment
```bash
source venv/bin/activate
```
---
### Usage
## Step 1 — Update Data
Download the latest historical results and upcoming fixtures before running the model:
```bash
python src/update_data.py
```
## Step 2 — Run the Main Pipeline
Runs tuning/loading, training/loading, evaluation, betting simulation and upcoming matchday predictions:
```bash
python main.py
```
## Step 3 — Predict a Custom Match
Run from the project root:
```bash
python -m src.predict_match
```
At the end of execution the system prints:
Aggregate evaluation metrics
Per-league evaluation metrics
Value betting simulation results
Upcoming matchday picks for all supported leagues


## Methodology Overview
```text
Raw Data
   │
   ├─► Elo Ratings
   │
   ├─► Poisson Team Strengths (time-weighted)
   │        │
   │        └─► Expected Goals (λ_home, λ_away)
   │                    │
   │                    └─► Dixon-Coles Scoreline Probabilities
   │
   ├─► Market Odds Probabilities
   │
   ├─► Team Momentum & Form History
   │
   └─► Meta Features
            │
            ├─► XGBoost Meta-Model
            ├─► MLP Meta-Model
            └─► Ensemble Blender
                    │
            Final Match Probabilities
                    │
          Value Bet Detection / Matchday Picks
```

## Evaluation

The system is evaluated on a held-out test set using:
Log Loss (NLL) — Main metric for probability quality. Lower is better.
Brier Score — Mean squared probability error. Lower is better.
ECE (Expected Calibration Error) — Measures calibration quality. Lower is better.
Accuracy — Useful as a secondary classification metric, but not the primary model-selection criterion.
Because this is a probabilistic prediction system, LogLoss, Brier and ECE are more important than plain accuracy.

Data Sources
Historical match results & odds: football-data.co.uk
Future fixtures: fixturedownload.com
---
Notes
Upcoming fixtures are read from dedicated fixture files, not inferred from historical result files alone.
The system predicts only the current / next available matchday window to avoid jumping too far into the future.
For custom match prediction, always run `predict_match.py` as a module from the project root:
```bash
  python -m src.predict_match
  ```
---
Thesis Context
This project is part of a diploma thesis on football match outcome prediction using:
classical statistical modelling,
machine learning meta-models,
ensemble methods,
and deep learning baselines.
The project has evolved from a single-file prototype into a more modular architecture to improve maintainability, experimentation and reproducibility.
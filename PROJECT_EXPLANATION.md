# Project Explanation

This document explains the project in plain language. It is meant for understanding the codebase and for explaining the thesis work without needing to read every Python file first.

For a focused explanation of which feature groups matter and which ones are mainly kept for ablation, see `FEATURE_IMPACT.md`.

For the betting target question, including why 60-70% accuracy usually requires a narrower task and how ROI is now tested with validation-locked filters, see `BETTING_TARGETS.md`.

## What This Project Does

The project predicts football match outcomes for the top five European leagues:

- England
- Spain
- Italy
- Germany
- France

For each match, it estimates the probability of:

- home win
- draw
- away win

It also checks whether the model finds value bets compared with bookmaker odds.

The important point is this: the project is not just trying to guess winners. It is trying to produce probabilities that are better calibrated than a baseline model and competitive with the betting market.

## The Main Idea

Football is difficult to predict because the bookmaker market is already strong. A simple model can look good by guessing favourites, but that does not mean it is useful for betting.

This project therefore compares several sources of information:

- team strength from historical results
- recent team form
- Elo ratings
- expected goals
- bookmaker odds
- market movement
- match statistics such as shots, corners, cards, and rest days

The pipeline then tests whether machine learning can combine those signals better than the market alone.

## The Pipeline In Simple Terms

The full flow is:

```text
Raw football data
  -> clean team names and dates
  -> add odds and match statistics
  -> add Understat expected goals
  -> build pre-match features
  -> run base Poisson/Elo model
  -> convert bookmaker odds into market probabilities
  -> train meta-models
  -> evaluate predictions on future matches
  -> simulate value betting
```

Each prediction is made using information available before that match. This is important because using future information would make the results unrealistic.

## Main Files

### Entry Points

- `main.py`
  Runs the full training, evaluation, reporting, and upcoming-picks pipeline.

- `predict_match.py`
  Runs the custom match prediction command-line tool.

- `src/update_data.py`
  Downloads football-data.co.uk result files and future fixture files.

- `src/update_understat.py`
  Downloads Understat expected-goals data and writes `data/external/understat_matches.csv`.

### Core Pipeline Files

- `src/data_processing.py`
  Loads raw league CSVs, cleans columns, parses dates, chooses odds, adds match statistics, and attaches Understat xG.

- `src/understat_data.py`
  Handles Understat-specific loading, team-name normalization, and merging Understat rows onto the main football-data rows.

- `src/state_builder.py`
  Builds the pre-match feature state. This is where rolling recent shots, xG, form, rest days, and market movement are calculated.

- `src/poisson_model.py`
  Builds the base football model. It estimates expected goals for both teams and converts those into home/draw/away probabilities.

- `src/elo.py`
  Maintains Elo ratings for teams over time.

- `src/feature_builder.py`
  Defines the final feature columns used by the meta-models.

- `src/trainer.py`
  Orchestrates the whole experiment: data loading, tuning, training, evaluation, ablations, reports, and saved artifacts.

- `src/evaluation.py`
  Contains evaluation and betting simulation logic.

## Data Sources

The project uses three main data sources:

```text
football-data.co.uk
  Historical match results, odds, shots, cards, corners, and other match stats.

fixturedownload.com
  Future fixture lists.

understat.com
  Expected goals, non-penalty expected goals, and expected points.
```

The raw data lives under:

```text
data/raw/
data/external/
```

The merged view with Understat is regenerated at:

```text
artifacts/merged_with_understat_all_leagues.csv
```

## What Understat Adds

Understat adds expected-goals information:

- `home_understat_xg`
- `away_understat_xg`
- `home_understat_npxg`
- `away_understat_npxg`
- `home_understat_xpts`
- `away_understat_xpts`

The model does not directly use the current match's xG to predict that same match. Instead, it builds rolling pre-match features, such as:

- average recent xG for the home team
- average recent xG for the away team
- difference between recent xG levels
- recent non-penalty xG
- recent expected points

This matters because xG for a match is only known after the match. Using the current match's xG before the match would be data leakage.

The merge now suppresses Understat values on rows marked as unplayed, so future fixtures do not accidentally receive post-match xG values.

## Model Types

### Base Model

The base model is a football-specific statistical model:

- Elo estimates team strength.
- A Poisson model estimates expected goals.
- Dixon-Coles adjustment improves low-score football probabilities.

This produces three probabilities:

```text
P(home win), P(draw), P(away win)
```

### Market Model

Bookmaker odds are converted into implied probabilities.

This is a very strong benchmark. If the machine learning model cannot beat the market on probability quality, that is not surprising. Betting markets are efficient because many people and algorithms are already competing there.

### Meta Models

The project trains models that learn from both:

- the base model probabilities
- the market probabilities
- extra features such as Elo, xG, form, rest, shots, and odds movement

The main meta-models are:

- XGBoost
- logistic regression
- MLP neural network
- blended ensemble

## Train, Validation, And Test Split

The active final config is in `src/config.py`.

Current final experiment:

```text
experiment_name = final_market_xg_comparison
train_cut = 2024-07-01
test_cut = 2025-07-01
```

The code uses time-based splitting:

```text
Before 2024-07-01:
  Used for fitting base league parameters and historical state.

2024-07-01 to 2025-07-01:
  Used as the validation/meta-training period.

From 2025-07-01 onward:
  Used as the test period.
```

This is much better than random train/test splitting because football data is chronological. A model should be tested on matches that happen after the data it was trained on.

## Evaluation Metrics

### Accuracy

Accuracy is the percentage of matches where the most likely predicted result was correct.

For football, accuracy is limited because there are three classes and draws are difficult. A model around 52-54% accuracy can still be meaningful.

Accuracy alone is not enough.

### Log Loss

Log loss measures probability quality.

Lower is better.

This is more important than accuracy because a betting model needs good probabilities, not just a correct top pick.

Example:

```text
Prediction A:
  Home 90%, Draw 5%, Away 5%

Prediction B:
  Home 45%, Draw 30%, Away 25%
```

If the match is not a home win, Prediction A is punished heavily. That is why log loss rewards calibrated probabilities.

### ROI

ROI comes from the betting simulation.

It answers:

```text
If we bet only when the model thinks the odds are good value, what would have happened?
```

ROI is noisy. A positive ROI in one test period is promising, but it is not proof that the model will make money in real betting.

The pipeline now adds validation-locked bet selection reports. These choose betting filters on the validation period first, then apply the chosen rules to the later test period. This is stricter than choosing thresholds after looking at test ROI.

## Current Saved Result Snapshot

The latest saved final summary currently shows roughly:

```text
Market odds:
  logloss around 0.975
  accuracy around 53.7%

Base model:
  logloss around 0.995
  accuracy around 51.5%

Meta model:
  logloss around 0.979
  accuracy around 53.6%
  simulated ROI slightly positive

Logistic regression:
  logloss around 0.978
  accuracy around 53.0%
  simulated ROI slightly positive

MLP:
  logloss around 0.980
  accuracy around 52.3%
  simulated ROI slightly positive
```

Interpretation:

```text
The market is still the best probability benchmark by log loss.
The learned models can be close to the market.
Some learned models show small positive simulated ROI.
The base Poisson/Elo model alone is weaker than the market.
```

Important: after refreshing data, rerun `python main.py` before quoting final numbers in the thesis. Saved artifacts can become stale when the underlying data changes.

## Is The Model Efficient?

There are two different meanings of "efficient."

### Code Efficiency

The code is reasonably modular and practical:

- data loading is separated from feature building
- model training is separated from evaluation
- artifacts are saved
- feature ablations are included
- tests cover important behavior

It is not a tiny script anymore. It is a real experiment pipeline.

Potential code improvements:

- add more documentation
- add more unit tests around date leakage and betting simulation
- reduce duplicated experiment artifacts
- make result tables easier to read
- add one command that refreshes all data and reruns the final experiment

### Prediction Efficiency

The model is not clearly better than the betting market on probability quality.

That does not mean the project failed. In football betting, the market is the hardest baseline. Showing that the model approaches the market and that extra features can improve some setups is already a defensible thesis result.

The honest conclusion is:

```text
The system improves over a standalone statistical model, but it does not consistently outperform bookmaker-implied probabilities by log loss.
```

For thesis purposes, that is a strong and realistic finding.

## What The Feature Ablations Mean

Feature ablation means training/evaluating the model with different groups of features removed or included.

Examples:

- `market_only`
  Uses bookmaker odds only.

- `no_market`
  Removes bookmaker odds to see what the model can do without the market.

- `no_understat_xg`
  Removes Understat xG features.

- `market_plus_understat_xg`
  Uses market probabilities plus Understat rolling xG features.

- `market_context_plus_understat_xg`
  Uses market probabilities, odds movement/context, and Understat features.

This helps answer:

```text
Which data sources actually help?
```

The current results suggest:

- market odds are the strongest single signal
- Understat features help some learned models
- removing the market hurts performance
- adding too many features does not always improve results

## How To Run The Project

From the repository root:

```powershell
python src/update_data.py
python src/update_understat.py
python main.py
```

To run the tests:

```powershell
python -m unittest tests.test_core_behaviors
```

To predict a custom match:

```powershell
python predict_match.py
```

## What To Say In The Thesis

A defensible thesis explanation would be:

```text
This project builds a football match prediction and value-betting evaluation system for the top five European leagues. It combines a domain-specific Poisson/Elo model with market-implied probabilities and additional engineered features, including recent form, rest days, match statistics, market movement, and Understat expected-goals indicators.

The evaluation uses chronological train/validation/test splits to reduce look-ahead bias. Results are compared against bookmaker-implied probabilities, which serve as a strong market-efficiency baseline. The learned models improve over the standalone statistical model, but the betting market remains difficult to beat by log loss. Some meta-model configurations show small positive simulated ROI, but this is treated as exploratory rather than proof of deployable profitability.
```

## What Not To Claim

Do not claim:

```text
The model guarantees profit.
The model clearly beats bookmakers.
The model is 85% or 93% accurate like older papers.
Accuracy alone proves success.
Current ROI proves future profitability.
```

Better claims:

```text
The model is competitive with the market baseline.
The market remains the strongest probability benchmark.
The extra feature groups can improve selected learned models.
The system provides a reproducible framework for evaluating football prediction and betting strategies.
```

## Known Limitations

The main limitations are:

- betting simulation is not the same as real betting execution
- bookmaker odds are highly efficient and hard to beat
- odds availability varies across leagues and seasons
- Understat starts in 2014, so 2012-2014 has no xG data
- player injuries, lineups, and team news are not fully modeled
- future fixture odds may be missing
- ROI can change a lot depending on the chosen test period

## Best Next Improvements

The most useful next improvements are:

1. Rerun the final experiment after every data refresh.
2. Add a final thesis table that clearly compares base, market, meta, logreg, MLP, and ensemble.
3. Add confidence intervals or bootstrap checks for ROI.
4. Add calibration plots or tables for probability quality.
5. Add a one-command script for:

```text
update football-data
update Understat
run tests
run final experiment
export final tables
```

## Short Version

The project is a real football prediction experiment pipeline.

It is strongest as a thesis system for comparing:

- statistical football models
- bookmaker market probabilities
- machine learning meta-models
- expected-goals features
- betting simulation

The honest result is not "we beat the market easily."

The honest result is:

```text
The model improves on a standalone football-statistical baseline and becomes competitive with market odds, but the market remains very difficult to outperform reliably.
```

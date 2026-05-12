# Betting Targets And Selection Reports

This document explains how the project now checks whether goals such as 60-70% accuracy or 10-20% ROI are realistic.

## Main Point

For all 1X2 matches, 60-70% accuracy is not a realistic target. The market and the current models are usually around the low-to-mid 50% range because draws and close matches are hard.

The only honest way to reach higher accuracy is to narrow the task:

- bet only high-confidence matches
- use easier markets such as double chance or draw-no-bet
- accept lower odds and fewer bets

The only honest way to test high ROI is to lock betting rules on validation data and then evaluate them on later test data.

## New Reports

Running `python main.py` now writes these extra files:

```text
artifacts/final_probability_quality_final_market_xg_comparison.csv
artifacts/final_bet_selector_final_market_xg_comparison.csv
artifacts/final_bet_buckets_final_market_xg_comparison.csv
artifacts/final_alternative_markets_final_market_xg_comparison.csv
artifacts/final_data_enrichment_audit_final_market_xg_comparison.csv
```

## What Each Report Means

### Probability Quality

This report compares validation and test performance using:

- log loss
- Brier score
- calibration error
- accuracy
- average confidence

This is the main place to judge probability quality. For betting, log loss and calibration matter more than raw accuracy.

### Validation-Locked Bet Selector

This report searches betting filters only on the validation period.

It tests combinations of:

- minimum model edge
- minimum model probability
- odds range

It marks a strategy as robust only if it has:

- enough validation bets
- positive validation ROI
- positive ROI in both chronological validation halves

Then the chosen filter is applied to the test period.

This prevents choosing a threshold because it happened to work on the final test set.

### Bet Buckets

This report explains where profit or losses come from:

- league
- predicted outcome
- odds bucket
- edge bucket
- confidence bucket

Use this to see whether a good ROI comes from broad behavior or from one narrow unstable segment.

### Alternative Markets

This report checks easier markets:

- double chance
- draw-no-bet
- over/under 2.5
- Asian handicap coverage

Double chance and draw-no-bet are accuracy-only audits because the current repo does not load real odds for those markets.

Over/under 2.5 can be simulated because the football-data files include O/U 2.5 odds in many seasons.

Asian handicap currently has line coverage, but handicap odds are not loaded yet.

### Data Enrichment Audit

This report shows which stronger real-world data is available now and which is missing.

Currently available:

- opening 1X2 odds
- closing 1X2 odds
- over/under 2.5 odds
- Asian handicap line

Currently missing:

- confirmed lineups
- injuries
- suspensions
- manager changes
- live odds snapshots
- weather

Those missing sources need external providers before they can be modeled honestly.

## How To Interpret 60-70% Accuracy

If double chance reaches 60-70%+ accuracy, that does not mean the original 1X2 model became 70% accurate. It means the task became easier because two outcomes can win.

A defensible thesis statement would be:

```text
High headline accuracy is achievable only under selective or simplified betting markets. For the full 1X2 task, probability quality and calibration are more meaningful than raw accuracy.
```

## How To Interpret 10-20% ROI

A 10-20% ROI on one slice is not enough. It should only be treated seriously if it:

- was selected on validation data
- survives the test period
- appears across more than one league or season
- is not caused by a tiny number of long-shot wins
- preferably beats closing-line value using pre-close odds

The current simulation mostly uses closing odds, so ROI is useful as an experiment but not proof of deployable betting profit.

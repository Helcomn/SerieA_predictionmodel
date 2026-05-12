# Feature Impact Guide

This document explains the feature set in a smaller, more understandable way.

The project currently defines 44 feature columns, but you should not think about them as 44 separate ideas. They are better understood as feature groups.

## Short Answer

The final thesis story should not be:

```text
We used 44 features.
```

That sounds uncontrolled and hard to defend.

The better story is:

```text
We evaluated seven groups of football prediction signals: base model probabilities, bookmaker market probabilities, team-strength context, recent form and momentum, market context, Understat expected-goals indicators, and local match statistics. Feature ablation showed that market probabilities and Understat-style quality features were the most useful, while local match-stat features such as shots, corners, and cards did not consistently improve performance.
```

## Feature Groups

### 1. Base Model Probabilities

Columns:

```text
model_logit_home
model_logit_draw
model_logit_away
```

Meaning:

These come from the internal Poisson/Elo football model.

Decision:

```text
Keep for comparison and model stacking.
```

Impact:

The base model alone is weaker than the market, but it is important because it shows what the football-statistical model can do without relying fully on bookmaker odds.

It is also useful for thesis explanation.

## 2. Market Probabilities

Columns:

```text
market_logit_home
market_logit_draw
market_logit_away
```

Meaning:

These are bookmaker odds converted into probabilities.

Decision:

```text
Keep. This is the strongest feature group.
```

Impact:

The market is the hardest benchmark to beat. In the latest saved results, market-only or market-heavy models are near the top by log loss.

This means bookmaker odds carry a lot of information.

## 3. Team Strength And Match Context

Columns:

```text
elo_diff
total_xg
xg_diff
rest_home
rest_away
rest_diff
```

Meaning:

These describe team strength, expected goal levels from the base model, and rest days.

Decision:

```text
Keep as core context, but do not overstate their standalone impact.
```

Impact:

These features help explain the model and support non-market prediction. They are not always enough to beat the market.

Rest days are plausible but not clearly a major driver in the saved ablations.

## 4. Form And Momentum

Columns:

```text
mom_home
mom_away
mom_diff
form_home
form_away
form_diff
```

Meaning:

Momentum is recent Elo movement.

Form is recent points won.

Decision:

```text
Keep as understandable football context.
```

Impact:

These are useful for the model narrative. They are intuitive and defensible.

They should be discussed as supporting features, not as the main reason the model works.

## 5. Local Match Statistics

Columns:

```text
shots_for_home_5
shots_for_away_5
shots_for_diff_5
sot_for_home_5
sot_for_away_5
sot_for_diff_5
corners_for_home_5
corners_for_away_5
corners_for_diff_5
cards_home_5
cards_away_5
cards_diff_5
```

Meaning:

Recent shots, shots on target, corners, and cards.

Decision:

```text
De-emphasize. Keep for ablation, but do not make these part of the main thesis claim.
```

Impact:

The saved ablations do not show consistent improvement from these features. In several cases, adding local stats makes performance worse.

Practical explanation:

These features sound useful, but they may be noisy, league-dependent, or already indirectly priced into bookmaker odds.

Thesis wording:

```text
Local match-stat features were tested but did not consistently improve out-of-sample log loss, suggesting that they added noise or duplicated information already captured by the market and team-strength features.
```

## 6. Market Context

Columns:

```text
market_move_home
market_move_draw
market_move_away
ou25_over_prob
ah_line
```

Meaning:

These describe extra betting-market information:

- how odds moved from opening to closing
- over/under 2.5 goal expectation
- Asian handicap line

Decision:

```text
Keep as a compact market-context group.
```

Impact:

Market context is often competitive. It is smaller and more defensible than the local-stat block.

It should be treated as market-derived information rather than independent football knowledge.

## 7. Understat Expected-Goals Features

Columns:

```text
understat_xg_for_home_5
understat_xg_for_away_5
understat_xg_diff_5
understat_npxg_for_home_5
understat_npxg_for_away_5
understat_npxg_diff_5
understat_xpts_home_5
understat_xpts_away_5
understat_xpts_diff_5
```

Meaning:

These measure recent chance quality:

- expected goals
- non-penalty expected goals
- expected points

Decision:

```text
Keep. This is one of the most important non-trivial feature groups.
```

Impact:

Understat features give the thesis a stronger football-analytics contribution than using only raw goals and bookmaker odds.

The latest saved model selections include Understat in the XGBoost model, and MLP ablations improve when Understat is included.

Important:

These are rolling pre-match features. The model should not use current-match xG before the match.

## Latest Saved Evidence

The latest saved ablation results show:

```text
XGBoost:
  market_plus_context               logloss 0.980261
  market_only                       logloss 0.980840
  market_context_plus_understat_xg  logloss 0.981005
  market_plus_understat_xg          logloss 0.982657
  core_plus_stats                   logloss 0.985229
  no_market                         logloss 0.995249

MLP:
  market_context_plus_understat_xg  logloss 0.984954
  market_plus_understat_xg          logloss 0.986233
  core_18                           logloss 0.987802
  all_features                      logloss 0.989737
  core_plus_stats                   logloss 0.998164
  no_understat_xg                   logloss 1.008213
```

Interpretation:

```text
The market matters most.
Understat features are useful and defensible.
All features together are not best.
Local stats are not clearly helpful.
Removing the market hurts a lot.
```

Note:

These are saved artifact results. After data refreshes, rerun `python main.py` before quoting exact numbers in the thesis.

## Recommended Final Feature Story

For the thesis, reduce the feature discussion to this:

```text
The feature set was organized into seven groups:

1. base model probabilities
2. bookmaker market probabilities
3. team-strength context
4. form and momentum
5. local match statistics
6. market context
7. Understat expected-goals indicators
```

Then say:

```text
Feature ablation showed that bookmaker odds and Understat expected-goals indicators were the most useful groups. Local match-stat features were retained for comparison but did not consistently improve out-of-sample performance.
```

## Practical Keep / Drop Table

| Group | Keep In Code? | Emphasize In Thesis? | Reason |
|---|---:|---:|---|
| Market probabilities | Yes | Yes | Strongest benchmark and signal |
| Understat xG/xPTS | Yes | Yes | Useful football-quality signal |
| Base model probabilities | Yes | Yes | Shows value of Poisson/Elo baseline |
| Form and momentum | Yes | Medium | Intuitive supporting context |
| Elo / base xG context | Yes | Medium | Useful model context |
| Market context | Yes | Medium | Compact and often helpful |
| Local stats | Yes for ablation | Low | No consistent performance gain |
| Rest days | Yes for now | Low/Medium | Plausible but not clearly decisive |

## Should We Delete Features?

Not yet.

Better approach:

```text
Keep all features in code so the ablation evidence remains reproducible.
Use a smaller recommended feature set for final reporting.
Explain that some feature groups were tested and rejected.
```

Deleting the local-stat features now would make it harder to show that they were tested.

## Recommended Final Model Framing

Use this wording:

```text
Although the implementation computes a broad set of 44 candidate columns, these correspond to seven interpretable feature groups. The final analysis focuses on the feature groups that were most effective in ablation testing, especially bookmaker-implied probabilities and Understat expected-goals indicators. Less effective groups, such as recent shots, corners, and cards, are included in the ablation study but are not emphasized as part of the final recommended model.
```


from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "model_logit_home", "model_logit_draw", "model_logit_away",
    "market_logit_home", "market_logit_draw", "market_logit_away",
    "elo_diff", "total_xg", "xg_diff", "mom_home", "mom_away", "mom_diff",
]


def market_probs_from_odds_row(odds_h, odds_d, odds_a):
    if not (np.isfinite(odds_h) and np.isfinite(odds_d) and np.isfinite(odds_a)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    if odds_h <= 1.0001 or odds_d <= 1.0001 or odds_a <= 1.0001:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    inv = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a], dtype=float)
    s = inv.sum()
    if s <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return inv / s


def safe_logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


def ensure_market_probs(model_probs: np.ndarray, market_probs: np.ndarray) -> np.ndarray:
    fixed = np.asarray(market_probs, dtype=float).copy()
    model_probs = np.asarray(model_probs, dtype=float)
    for i in range(len(fixed)):
        if not np.isfinite(fixed[i]).all():
            fixed[i] = model_probs[i]
    return fixed


def build_meta_features(model_probs: np.ndarray, market_probs: np.ndarray, aux: np.ndarray) -> np.ndarray:
    model_probs = np.asarray(model_probs, dtype=float)
    market_fixed = ensure_market_probs(model_probs, market_probs)
    aux = np.asarray(aux, dtype=float)

    X = []
    for i in range(len(model_probs)):
        pm = model_probs[i]
        pk = market_fixed[i]
        feats = [
            safe_logit(pm[0]), safe_logit(pm[1]), safe_logit(pm[2]),
            safe_logit(pk[0]), safe_logit(pk[1]), safe_logit(pk[2]),
        ]
        feats.extend(aux[i].tolist())
        X.append(feats)
    return np.array(X, dtype=float)


def build_single_feature_vector(model_probs, market_probs, *, elo_h, elo_a, lam_h, lam_a, mom_h, mom_a):
    mom_diff = mom_h - mom_a
    aux = np.array([[(elo_h - elo_a) / 400.0, lam_h + lam_a, lam_h - lam_a, mom_h, mom_a, mom_diff]], dtype=float)
    return build_meta_features(np.asarray([model_probs], dtype=float), np.asarray([market_probs], dtype=float), aux)


def time_split_val(val_df: pd.DataFrame):
    val_sorted = val_df.sort_values("date").reset_index(drop=True)
    mid = len(val_sorted) // 2
    return val_sorted.iloc[:mid].copy(), val_sorted.iloc[mid:].copy()

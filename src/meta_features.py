import numpy as np
import pandas as pd

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

def build_meta_features(p_model, p_mkt, aux):
    p_mkt_fixed = p_mkt.copy()
    for i in range(len(p_mkt_fixed)):
        if not np.isfinite(p_mkt_fixed[i]).all():
            p_mkt_fixed[i] = p_model[i]

    X = []
    for i in range(len(p_model)):
        pm = p_model[i]
        pk = p_mkt_fixed[i]
        # Κρατάμε τα βασικά logits
        feats = [
            safe_logit(pm[0]), safe_logit(pm[1]), safe_logit(pm[2]),
            safe_logit(pk[0]), safe_logit(pk[1]), safe_logit(pk[2]),
        ]
        # Προσθέτουμε δυναμικά όσα aux features μας έρθουν (π.χ. Momentum)
        feats.extend(aux[i])
        X.append(feats)
    return np.array(X, dtype=float)

def time_split_val(val_df: pd.DataFrame):
    val_sorted = val_df.sort_values("date").reset_index(drop=True)
    mid = len(val_sorted) // 2
    return val_sorted.iloc[:mid].copy(), val_sorted.iloc[mid:].copy()
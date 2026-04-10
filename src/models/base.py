from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss

from src.calibration import fit_temperature, temperature_scale_probs
from src.elo import compute_elo_ratings
from src.poisson_model import fit_team_strengths, fit_team_strengths_weighted, match_outcome_probs, predict_lambdas
from src.state_builder import streaming_block_probs_home_away


def tune_league_params(train_fit, val, full_played_df):
    Ks = [40, 50, 60, 70]
    home_advs = [60, 80, 100, 110]
    betas = [0.10, 0.11, 0.12, 0.13]
    decays = [0.0005, 0.001, 0.002, 0.003]

    best = None
    print("\n--- Elo & Beta Tuning ---")
    import pandas as pd
    from src.poisson_model import apply_elo_to_lambdas

    for K in Ks:
        for ha in home_advs:
            for b in betas:
                elo_pairs = compute_elo_ratings(train_fit, K=K, home_adv=ha, use_margin=True)
                tmp = train_fit.copy()
                tmp["elo_home"], tmp["elo_away"] = zip(*elo_pairs)
                l_avg_h, l_avg_a, att, dfn = fit_team_strengths(tmp)

                full_tmp = pd.concat([train_fit, val], ignore_index=True).sort_values("date")
                elo_full = compute_elo_ratings(full_tmp, K=K, home_adv=ha, use_margin=True)
                full_tmp["elo_home"], full_tmp["elo_away"] = zip(*elo_full)
                val_part = full_tmp.iloc[len(train_fit):]

                probs = []
                y = []
                for _, row in val_part.iterrows():
                    lh, la = predict_lambdas(row["home_team"], row["away_team"], l_avg_h, l_avg_a, att, dfn)
                    lh, la = apply_elo_to_lambdas(lh, la, row["elo_home"], row["elo_away"], beta=b)
                    probs.append(match_outcome_probs(lh, la))
                    if row["home_goals"] > row["away_goals"]:
                        y.append(0)
                    elif row["home_goals"] == row["away_goals"]:
                        y.append(1)
                    else:
                        y.append(2)

                ll = log_loss(np.array(y), np.array(probs))
                if best is None or ll < best[0]:
                    best = (ll, K, ha, b)

    _, best_K, best_ha, best_beta = best
    print(f"Best Config: K={best_K}, ha={best_ha}, beta={best_beta}")

    print("--- Tuning Time Decay ---")
    best_decay, best_decay_ll = None, float("inf")
    y_val = np.array([0 if r["home_goals"] > r["away_goals"] else 1 if r["home_goals"] == r["away_goals"] else 2 for _, r in val.iterrows()], dtype=int)

    elo_full = compute_elo_ratings(
        full_played_df[full_played_df["date"] < val["date"].max() + np.timedelta64(1, "D")],
        K=best_K,
        home_adv=best_ha,
        use_margin=True,
    )
    tmp_df = full_played_df[full_played_df["date"] < val["date"].max() + np.timedelta64(1, "D")].copy()
    tmp_df["elo_home"], tmp_df["elo_away"] = zip(*elo_full)
    val_part = tmp_df[tmp_df["date"].isin(val["date"])].copy()

    from src.poisson_model import apply_elo_to_lambdas
    for d in decays:
        l_avg_h, l_avg_a, att, dfn = fit_team_strengths_weighted(train_fit, decay=d)
        probs = []
        for _, row in val_part.iterrows():
            lh, la = predict_lambdas(row["home_team"], row["away_team"], l_avg_h, l_avg_a, att, dfn)
            lh, la = apply_elo_to_lambdas(lh, la, row["elo_home"], row["elo_away"], beta=best_beta)
            probs.append(match_outcome_probs(lh, la))
        ll = log_loss(y_val, np.array(probs))
        if ll < best_decay_ll:
            best_decay_ll, best_decay = ll, d

    print(f"Best Decay: {best_decay}")
    print("--- Joint Tuning rho + Temperature ---")
    rho_grid = np.arange(-0.2, 0.21, 0.02)
    best_joint = None
    for rho in rho_grid:
        print(f"  rho={rho:.2f}")
        val_probs_raw, y_val_stream, _, _, _ = streaming_block_probs_home_away(
            val, full_played_df,
            beta=best_beta, rho=float(rho), decay=best_decay,
            K=best_K, home_adv=best_ha,
        )
        T = fit_temperature(val_probs_raw, y_val_stream)
        val_probs_cal = temperature_scale_probs(val_probs_raw, T)
        ll = log_loss(y_val_stream, val_probs_cal)
        if best_joint is None or ll < best_joint[0]:
            best_joint = (ll, float(rho), float(T))

    _, best_rho, best_T = best_joint
    print(f"Best rho={best_rho}, T={best_T}")
    return {"K": int(best_K), "ha": int(best_ha), "beta": float(best_beta), "decay": float(best_decay), "rho": float(best_rho), "T": float(best_T)}

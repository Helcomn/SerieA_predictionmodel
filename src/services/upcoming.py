from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration import temperature_scale_probs
from src.data_loader import load_league_data
from src.feature_builder import MLP_DEFAULT_COLS, build_meta_features, feature_indices, market_probs_from_odds_row, ensure_market_probs
from src.fixtures import get_current_or_next_matchday_fixtures
from src.models.meta import blend_probabilities
from src.poisson_model import top_k_scorelines_dc
from src.state_builder import build_league_state, compute_match_components, compute_pre_match_extra_features


def generate_upcoming_matchday_picks(
    leagues,
    league_best_params,
    meta_model,
    meta_cfg,
    mlp_model,
    mlp_cfg,
    *,
    max_window_days=4,
    pick_model="ensemble",
):
    print("\n" + "=" * 70)
    print(f"=== CURRENT / NEXT MATCHDAY PICKS ({pick_model.upper()}) ===")
    print("=" * 70)

    upcoming_rows = []
    for league in leagues:
        params = league_best_params.get(league)
        if params is None:
            continue
        df_league_all = load_league_data(league).sort_values("date").reset_index(drop=True)
        fixtures, matchday_start = get_current_or_next_matchday_fixtures(df_league_all, max_window_days=max_window_days)
        if fixtures.empty or matchday_start is None:
            print(f"{league.upper()}: No current/upcoming matchday fixtures found.")
            continue
        print(f"\n{league.upper()} MATCHDAY starting: {matchday_start.date()}")

        played_df = df_league_all[(df_league_all["is_played"] == True) & (df_league_all["date"] < matchday_start)].copy()
        if played_df.empty:
            continue
        state = build_league_state(played_df, params)

        model_raw, market, aux, lambdas = [], [], [], []
        for _, row in fixtures.iterrows():
            past_matches = played_df[played_df["date"] < row["date"]]
            extra_aux = compute_pre_match_extra_features(row, past_matches)
            comp = compute_match_components(row["home_team"], row["away_team"], state, match_date=row["date"], extra_aux=extra_aux)
            model_raw.append(comp["probs"].tolist())
            market.append(market_probs_from_odds_row(row["odds_home"], row["odds_draw"], row["odds_away"]).tolist())
            aux.append(comp["aux"].tolist())
            lambdas.append([comp["lam_home"], comp["lam_away"]])

        model_raw = np.array(model_raw, dtype=float)
        market = np.array(market, dtype=float)
        aux = np.array(aux, dtype=float)
        lambdas = np.array(lambdas, dtype=float)

        model_probs = temperature_scale_probs(model_raw, params["T"])
        market_fixed = ensure_market_probs(model_probs, market)

        X_future = build_meta_features(model_probs, market_fixed, aux)
        xgb_cols = feature_indices(meta_cfg.get("feature_columns", [])) if meta_cfg is not None and meta_cfg.get("feature_columns") else list(range(X_future.shape[1]))
        future_meta_probs = meta_model.predict_proba(X_future[:, xgb_cols])
        mlp_cols = feature_indices(mlp_cfg.get("feature_columns", [])) if mlp_cfg is not None and mlp_cfg.get("feature_columns") else MLP_DEFAULT_COLS
        future_mlp_probs_raw = mlp_model.predict_proba(X_future[:, mlp_cols])
        future_mlp_probs = temperature_scale_probs(future_mlp_probs_raw, float(mlp_cfg["temperature"]))

        blend_cfg = params.get("_blend_cfg")
        if blend_cfg is None:
            future_ensemble_probs = future_meta_probs
        else:
            future_ensemble_probs = blend_probabilities(
                blend_cfg["weights"],
                {
                    "base": model_probs,
                    "market": market_fixed,
                    "xgb": future_meta_probs,
                    "mlp": future_mlp_probs,
                },
            )
        candidate_probs = {
            "base": model_probs,
            "market": market_fixed,
            "meta": future_meta_probs,
            "mlp": future_mlp_probs,
            "ensemble": future_ensemble_probs,
        }
        future_probs = candidate_probs.get(pick_model, future_ensemble_probs)
        for i, (_, row) in enumerate(fixtures.iterrows()):
            if not np.isfinite(future_probs[i]).all():
                future_probs[i] = future_meta_probs[i]

            pH, pD, pA = future_probs[i]
            lam_h, lam_a = lambdas[i]
            result_pick = ["H", "D", "A"][np.argmax([pH, pD, pA])]
            top_score = top_k_scorelines_dc(lam_h, lam_a, rho=params["rho"], k=1, max_goals=6)
            (hg, ag), score_prob = top_score[0]
            upcoming_rows.append({
                "League": league.upper(),
                "Date": row["date"].strftime("%Y-%m-%d"),
                "Home": row["home_team"],
                "Away": row["away_team"],
                "P(H)": round(float(pH), 3),
                "P(D)": round(float(pD), 3),
                "P(A)": round(float(pA), 3),
                "Pick": result_pick,
                "Score Pick": f"{hg}-{ag}",
                "Score Prob": round(float(score_prob), 3),
                "Value Bet": "-",
            })

    if not upcoming_rows:
        print("No current/upcoming matchday fixtures available.")
        return pd.DataFrame()
    picks_df = pd.DataFrame(upcoming_rows).sort_values(["Date", "League", "Home"]).reset_index(drop=True)
    print("\n")
    print(picks_df.to_string(index=False))
    return picks_df

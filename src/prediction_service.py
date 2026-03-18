import numpy as np
import pandas as pd

from src.data_processing import load_league_data
from src.fixtures import get_current_or_next_matchday_fixtures
from src.meta_features import market_probs_from_odds_row, build_meta_features
from src.poisson_model import (
    fit_team_strengths_home_away_weighted,
    predict_lambdas_home_away,
    apply_elo_to_lambdas,
    match_outcome_probs_dc,
    top_k_scorelines_dc,
)
from src.calibration import temperature_scale_probs
from src.tuning import blend_probabilities


def generate_upcoming_matchday_picks(
    leagues,
    league_best_params,
    meta_model,
    mlp_model,
    mlp_cfg,
    max_window_days=4,
):
    print("\n" + "=" * 70)
    print("=== CURRENT / NEXT MATCHDAY PICKS (ENSEMBLE) ===")
    print("=" * 70)

    upcoming_rows = []

    for league in leagues:
        params = league_best_params.get(league)
        if params is None:
            continue

        df_league_all = load_league_data(league).sort_values("date").reset_index(drop=True)
        fixtures, matchday_start = get_current_or_next_matchday_fixtures(
            df_league_all,
            max_window_days=max_window_days,
        )

        if fixtures.empty or matchday_start is None:
            print(f"{league.upper()}: No current/upcoming matchday fixtures found.")
            continue

        print(f"\n{league.upper()} MATCHDAY starting: {matchday_start.date()}")

        played_df = df_league_all[
            (df_league_all["is_played"] == True) &
            (df_league_all["date"] < matchday_start)
        ].copy()

        if played_df.empty:
            print(f"{league.upper()}: Not enough played history before selected matchday.")
            continue

        from src.elo import expected_score, match_result, margin_multiplier
        ratings = {}
        elo_history = {} # NEW: Ιστορικό για υπολογισμό Momentum

        def get_dynamic_init_local(current_ratings, default_init=1500.0):
            if len(current_ratings) >= 5:
                bottom_elos = sorted(current_ratings.values())[:3]
                return float(sum(bottom_elos) / len(bottom_elos))
            return float(default_init)

        def update_ratings_local(matches_batch, current_ratings, current_history):
            for _, m in matches_batch.iterrows():
                h, a = m["home_team"], m["away_team"]
                dyn_init = get_dynamic_init_local(current_ratings)
                r_h = current_ratings.get(h, dyn_init)
                r_a = current_ratings.get(a, dyn_init)

                if h not in current_ratings: current_ratings[h] = r_h
                if a not in current_ratings: current_ratings[a] = r_a
                if h not in current_history: current_history[h] = []
                if a not in current_history: current_history[a] = []

                exp_h = expected_score(r_h + params["ha"], r_a)
                s_h, s_a = match_result(int(m["home_goals"]), int(m["away_goals"]))
                mult = margin_multiplier(int(m["home_goals"]) - int(m["away_goals"]))

                new_r_h = r_h + (params["K"] * mult) * (s_h - exp_h)
                new_r_a = r_a + (params["K"] * mult) * (s_a - (1 - exp_h))

                current_ratings[h] = new_r_h
                current_ratings[a] = new_r_a
                current_history[h].append(new_r_h)
                current_history[a].append(new_r_a)

            return current_ratings, current_history

        def get_momentum_local(team, current_r, history, window=4):
            if team not in history or len(history[team]) < window:
                return 0.0
            return current_r - history[team][-window]

        ratings, elo_history = update_ratings_local(played_df, ratings, elo_history)

        l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
            played_df, decay=params["decay"]
        )

        fixture_model_probs_raw = []
        fixture_market_probs = []
        fixture_aux = []
        fixture_lambdas = []

        for _, row in fixtures.iterrows():
            ht, at = row["home_team"], row["away_team"]

            dyn_init = get_dynamic_init_local(ratings)
            elo_h = ratings.get(ht, dyn_init)
            elo_a = ratings.get(at, dyn_init)

            lam_h, lam_a = predict_lambdas_home_away(
                ht, at,
                l_avg_h, l_avg_a,
                att_h, def_h,
                att_a, def_a,
            )
            lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, elo_h, elo_a, beta=params["beta"])
            pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=params["rho"], max_goals=10)

            fixture_model_probs_raw.append([pH, pD, pA])
            fixture_market_probs.append(
                market_probs_from_odds_row(
                    row["odds_home"], row["odds_draw"], row["odds_away"]
                ).tolist()
            )
            
            # Υπολογισμός Momentum
            mom_h = get_momentum_local(ht, elo_h, elo_history, window=4) / 400.0
            mom_a = get_momentum_local(at, elo_a, elo_history, window=4) / 400.0
            mom_diff = mom_h - mom_a

            # Πλέον στέλνουμε τις 6 μεταβλητές!
            fixture_aux.append([
                (elo_h - elo_a) / 400.0,
                lam_h + lam_a,
                lam_h - lam_a,
                mom_h,
                mom_a,
                mom_diff
            ])
            fixture_lambdas.append([lam_h, lam_a])

        fixture_model_probs_raw = np.array(fixture_model_probs_raw, dtype=float)
        fixture_market_probs = np.array(fixture_market_probs, dtype=float)
        fixture_aux = np.array(fixture_aux, dtype=float)
        fixture_lambdas = np.array(fixture_lambdas, dtype=float)

        fixture_model_probs = temperature_scale_probs(fixture_model_probs_raw, params["T"])

        fixture_market_fixed = fixture_market_probs.copy()
        for i in range(len(fixture_market_fixed)):
            if not np.isfinite(fixture_market_fixed[i]).all():
                fixture_market_fixed[i] = fixture_model_probs[i]

        X_future = build_meta_features(fixture_model_probs, fixture_market_fixed, fixture_aux)
        future_meta_probs = meta_model.predict_proba(X_future)
        future_mlp_probs_raw = mlp_model.predict_proba(X_future)
        future_mlp_probs = temperature_scale_probs(
            future_mlp_probs_raw,
            float(mlp_cfg["temperature"]),
        )
        
        blend_cfg = params.get("_blend_cfg")
        if blend_cfg is None:
            future_probs = future_meta_probs
        else:
            future_probs = blend_probabilities(
                blend_cfg["weights"],
                {"base": fixture_model_probs, "market": fixture_market_fixed, "xgb": future_meta_probs, "mlp": future_mlp_probs},
            )

        for i, (_, row) in enumerate(fixtures.iterrows()):
            pH, pD, pA = future_probs[i]
            lam_h, lam_a = fixture_lambdas[i]
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
            })

    if len(upcoming_rows) == 0:
        print("No current/upcoming matchday fixtures available.")
        return pd.DataFrame()

    picks_df = pd.DataFrame(upcoming_rows)
    picks_df = picks_df.sort_values(["Date", "League", "Home"]).reset_index(drop=True)
    print("\n")
    print(picks_df.to_string(index=False))
    return picks_df
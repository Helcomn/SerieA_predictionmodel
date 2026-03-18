import numpy as np

from src.meta_features import market_probs_from_odds_row
from src.poisson_model import (
    fit_team_strengths_home_away_weighted,
    predict_lambdas_home_away,
    apply_elo_to_lambdas,
    match_outcome_probs_dc,
)

def streaming_block_probs_home_away(
    predict_df, full_df, beta, rho, decay, K, home_adv, init_rating=1500.0, max_goals=10
):
    from src.elo import expected_score, match_result, margin_multiplier

    probs_model = []
    probs_mkt = []
    y_true = []
    aux = []

    predict_df = predict_df.sort_values("date")
    full_df = full_df.sort_values("date")

    predict_dates = sorted(predict_df["date"].unique())
    if len(predict_dates) == 0:
        return (np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0, 3)), np.zeros((0, 3)))

    first_predict_date = predict_dates[0]
    ratings = {}
    elo_history = {} # NEW: Ιστορικό για τη Φόρμα (Momentum)

    history_matches = full_df[full_df["date"] < first_predict_date]

    def get_dynamic_init(current_ratings):
        if len(current_ratings) >= 5:
            bottom_elos = sorted(current_ratings.values())[:3]
            return float(sum(bottom_elos) / len(bottom_elos))
        return float(init_rating)

    def update_ratings(matches_batch, current_ratings, current_history):
        for _, m in matches_batch.iterrows():
            h, a = m["home_team"], m["away_team"]
            dyn_init = get_dynamic_init(current_ratings)
            r_h = current_ratings.get(h, dyn_init)
            r_a = current_ratings.get(a, dyn_init)
            
            if h not in current_ratings: current_ratings[h] = r_h
            if a not in current_ratings: current_ratings[a] = r_a
            
            if h not in current_history: current_history[h] = []
            if a not in current_history: current_history[a] = []

            exp_h = expected_score(r_h + home_adv, r_a)
            s_h, s_a = match_result(int(m["home_goals"]), int(m["away_goals"]))
            mult = margin_multiplier(int(m["home_goals"]) - int(m["away_goals"]))
            
            new_r_h = r_h + (K * mult) * (s_h - exp_h)
            new_r_a = r_a + (K * mult) * (s_a - (1 - exp_h))
            
            current_ratings[h] = new_r_h
            current_ratings[a] = new_r_a
            
            current_history[h].append(new_r_h)
            current_history[a].append(new_r_a)
            
        return current_ratings, current_history

    ratings, elo_history = update_ratings(history_matches, ratings, elo_history)

    def get_momentum(team, current_r, history, window=4):
        if team not in history or len(history[team]) < window:
            return 0.0
        return current_r - history[team][-window]

    for d in predict_dates:
        day_matches = predict_df[predict_df["date"] == d]
        past_matches = full_df[full_df["date"] < d]

        l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
            past_matches, decay=decay
        )

        for _, row in day_matches.iterrows():
            ht, at = row["home_team"], row["away_team"]
            dyn_init = get_dynamic_init(ratings)
            elo_h = ratings.get(ht, dyn_init)
            elo_a = ratings.get(at, dyn_init)

            lam_h, lam_a = predict_lambdas_home_away(
                ht, at, l_avg_h, l_avg_a, att_h, def_h, att_a, def_a
            )
            lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, elo_h, elo_a, beta=beta)

            pH, pD, pA = match_outcome_probs_dc(lam_h, lam_a, rho=rho, max_goals=max_goals)
            probs_model.append([pH, pD, pA])

            pm = market_probs_from_odds_row(row["odds_home"], row["odds_draw"], row["odds_away"])
            probs_mkt.append(pm.tolist())

            if row["home_goals"] > row["away_goals"]: y_true.append(0)
            elif row["home_goals"] == row["away_goals"]: y_true.append(1)
            else: y_true.append(2)

            # --- ΝΕΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ (FEATURE ENGINEERING) ---
            elo_diff = (elo_h - elo_a) / 400.0
            total_xg = lam_h + lam_a
            xg_diff = lam_h - lam_a
            
            mom_h = get_momentum(ht, elo_h, elo_history, window=4) / 400.0
            mom_a = get_momentum(at, elo_a, elo_history, window=4) / 400.0
            mom_diff = mom_h - mom_a
            
            # Τώρα στέλνουμε 6 βοηθητικές μεταβλητές αντί για 3!
            aux.append([elo_diff, total_xg, xg_diff, mom_h, mom_a, mom_diff])

        ratings, elo_history = update_ratings(day_matches, ratings, elo_history)

    return (
        np.array(probs_model, dtype=float),
        np.array(y_true, dtype=int),
        np.array(probs_mkt, dtype=float),
        np.array(aux, dtype=float),
    )
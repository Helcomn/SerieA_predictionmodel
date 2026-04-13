from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.elo import expected_score, match_result, margin_multiplier
from src.feature_builder import market_probs_from_odds_row
from src.poisson_model import (
    apply_elo_to_lambdas,
    fit_team_strengths_home_away_weighted,
    match_outcome_probs_dc,
    predict_lambdas_home_away,
)


@dataclass
class LeagueState:
    ratings: Dict[str, float]
    elo_history: Dict[str, list[float]]
    last_match_date: Dict[str, pd.Timestamp]
    points_history: Dict[str, list[float]]
    attack_home: dict
    defense_home: dict
    attack_away: dict
    defense_away: dict
    league_avg_home: float
    league_avg_away: float
    params: dict


def dynamic_init_rating(ratings: Dict[str, float], init_rating: float = 1500.0) -> float:
    if len(ratings) >= 5:
        bottom_elos = sorted(ratings.values())[:3]
        return float(sum(bottom_elos) / len(bottom_elos))
    return float(init_rating)


def update_elo_state(
    matches_batch: pd.DataFrame,
    ratings: Dict[str, float],
    elo_history: Dict[str, list[float]],
    last_match_date: Dict[str, pd.Timestamp],
    points_history: Dict[str, list[float]],
    *,
    K: float,
    home_adv: float,
    init_rating: float = 1500.0,
):
    for _, m in matches_batch.iterrows():
        h, a = m["home_team"], m["away_team"]
        dyn_init = dynamic_init_rating(ratings, init_rating)
        r_h = ratings.get(h, dyn_init)
        r_a = ratings.get(a, dyn_init)

        ratings.setdefault(h, r_h)
        ratings.setdefault(a, r_a)
        elo_history.setdefault(h, [])
        elo_history.setdefault(a, [])
        points_history.setdefault(h, [])
        points_history.setdefault(a, [])

        exp_h = expected_score(r_h + home_adv, r_a)
        s_h, s_a = match_result(int(m["home_goals"]), int(m["away_goals"]))
        mult = margin_multiplier(int(m["home_goals"]) - int(m["away_goals"]))

        new_r_h = r_h + (K * mult) * (s_h - exp_h)
        new_r_a = r_a + (K * mult) * (s_a - (1.0 - exp_h))

        ratings[h] = new_r_h
        ratings[a] = new_r_a
        elo_history[h].append(new_r_h)
        elo_history[a].append(new_r_a)
        if s_h == 1.0:
            points_history[h].append(3.0)
            points_history[a].append(0.0)
        elif s_h == 0.5:
            points_history[h].append(1.0)
            points_history[a].append(1.0)
        else:
            points_history[h].append(0.0)
            points_history[a].append(3.0)
        match_date = pd.Timestamp(m["date"])
        last_match_date[h] = match_date
        last_match_date[a] = match_date
    return ratings, elo_history, last_match_date, points_history


def get_team_momentum(team: str, current_rating: float, elo_history: Dict[str, list[float]], window: int = 4) -> float:
    if team not in elo_history or len(elo_history[team]) < window:
        return 0.0
    return (current_rating - elo_history[team][-window]) / 400.0


def get_recent_points_form(team: str, points_history: Dict[str, list[float]], window: int = 5) -> float:
    team_points = points_history.get(team, [])
    if not team_points:
        return 0.0
    recent = team_points[-window:]
    return float(sum(recent)) / (3.0 * len(recent))


def get_rest_days(team: str, match_date: pd.Timestamp, last_match_date: Dict[str, pd.Timestamp], default_days: int = 7, cap_days: int = 21) -> float:
    last_date = last_match_date.get(team)
    if last_date is None:
        return float(default_days) / 7.0
    days = max(0, int((pd.Timestamp(match_date) - pd.Timestamp(last_date)).days))
    days = min(days, cap_days)
    return float(days) / 7.0


def build_league_state(played_df: pd.DataFrame, params: dict) -> LeagueState:
    played_df = played_df.sort_values("date").reset_index(drop=True)
    l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(played_df, decay=params["decay"])

    ratings: Dict[str, float] = {}
    elo_history: Dict[str, list[float]] = {}
    last_match_date: Dict[str, pd.Timestamp] = {}
    points_history: Dict[str, list[float]] = {}
    update_elo_state(
        played_df,
        ratings,
        elo_history,
        last_match_date,
        points_history,
        K=params["K"],
        home_adv=params["ha"],
    )

    return LeagueState(
        ratings=ratings,
        elo_history=elo_history,
        last_match_date=last_match_date,
        points_history=points_history,
        attack_home=att_h,
        defense_home=def_h,
        attack_away=att_a,
        defense_away=def_a,
        league_avg_home=l_avg_h,
        league_avg_away=l_avg_a,
        params=params,
    )


def compute_match_components(home_team: str, away_team: str, state: LeagueState, match_date: pd.Timestamp | None = None):
    p = state.params
    dyn_init = dynamic_init_rating(state.ratings)
    elo_h = state.ratings.get(home_team, dyn_init)
    elo_a = state.ratings.get(away_team, dyn_init)
    if match_date is None:
        latest_seen = max(state.last_match_date.values()) if state.last_match_date else pd.Timestamp.today().normalize()
        match_date = latest_seen + pd.Timedelta(days=1)

    lam_h, lam_a = predict_lambdas_home_away(
        home_team,
        away_team,
        state.league_avg_home,
        state.league_avg_away,
        state.attack_home,
        state.defense_home,
        state.attack_away,
        state.defense_away,
    )
    lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, elo_h, elo_a, beta=p["beta"])
    probs = np.array(match_outcome_probs_dc(lam_h, lam_a, rho=p["rho"], max_goals=10), dtype=float)
    mom_h = get_team_momentum(home_team, elo_h, state.elo_history)
    mom_a = get_team_momentum(away_team, elo_a, state.elo_history)
    rest_h = get_rest_days(home_team, match_date, state.last_match_date)
    rest_a = get_rest_days(away_team, match_date, state.last_match_date)
    form_h = get_recent_points_form(home_team, state.points_history)
    form_a = get_recent_points_form(away_team, state.points_history)
    aux = np.array([
        (elo_h - elo_a) / 400.0,
        lam_h + lam_a,
        lam_h - lam_a,
        mom_h,
        mom_a,
        mom_h - mom_a,
        rest_h,
        rest_a,
        rest_h - rest_a,
        form_h,
        form_a,
        form_h - form_a,
    ], dtype=float)
    return {
        "elo_home": elo_h,
        "elo_away": elo_a,
        "lam_home": lam_h,
        "lam_away": lam_a,
        "probs": probs,
        "aux": aux,
        "mom_home": mom_h,
        "mom_away": mom_a,
        "rest_home": rest_h,
        "rest_away": rest_a,
        "form_home": form_h,
        "form_away": form_a,
    }


def streaming_block_probs_home_away(predict_df, full_df, beta, rho, decay, K, home_adv, init_rating=1500.0, max_goals=10):
    params = {"beta": beta, "rho": rho, "decay": decay, "K": K, "ha": home_adv}
    probs_model = []
    probs_mkt = []
    y_true = []
    aux = []
    raw_odds = []

    predict_df = predict_df.sort_values("date")
    full_df = full_df.sort_values("date")
    predict_dates = sorted(predict_df["date"].unique())
    if len(predict_dates) == 0:
        return (np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0, 3)), np.zeros((0, 12)), np.zeros((0, 3)))

    ratings = {}
    elo_history = {}
    last_match_date = {}
    points_history = {}
    history_matches = full_df[full_df["date"] < predict_dates[0]]
    update_elo_state(
        history_matches,
        ratings,
        elo_history,
        last_match_date,
        points_history,
        K=K,
        home_adv=home_adv,
        init_rating=init_rating,
    )

    for d in predict_dates:
        day_matches = predict_df[predict_df["date"] == d]
        past_matches = full_df[full_df["date"] < d]

        l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
            past_matches, decay=decay
    )
        state = LeagueState(
        ratings=ratings.copy(),
        elo_history={k: v[:] for k, v in elo_history.items()},
        last_match_date=last_match_date.copy(),
        points_history={k: v[:] for k, v in points_history.items()},
        attack_home=att_h,
        defense_home=def_h,
        attack_away=att_a,
        defense_away=def_a,
        league_avg_home=l_avg_h,
        league_avg_away=l_avg_a,
        params={**params, "K": K, "ha": home_adv},
    )

        for _, row in day_matches.iterrows():
            raw_odds.append([row["odds_home"], row["odds_draw"], row["odds_away"]])
            comp = compute_match_components(row["home_team"], row["away_team"], state, match_date=row["date"])
            probs_model.append(comp["probs"].tolist())
            probs_mkt.append(market_probs_from_odds_row(row["odds_home"], row["odds_draw"], row["odds_away"]).tolist())
            aux.append(comp["aux"].tolist())
            if row["home_goals"] > row["away_goals"]:
                y_true.append(0)
            elif row["home_goals"] == row["away_goals"]:
                y_true.append(1)
            else:
                y_true.append(2)

        update_elo_state(
            day_matches,
            ratings,
            elo_history,
            last_match_date,
            points_history,
            K=K,
            home_adv=home_adv,
            init_rating=init_rating,
        )

    return (
        np.array(probs_model, dtype=float),
        np.array(y_true, dtype=int),
        np.array(probs_mkt, dtype=float),
        np.array(aux, dtype=float),
        np.array(raw_odds, dtype=float),
    )

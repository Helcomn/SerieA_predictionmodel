import numpy as np
import pandas as pd


def labels_from_df(df: pd.DataFrame) -> np.ndarray:
    y = []
    for _, r in df.iterrows():
        if r["home_goals"] > r["away_goals"]:
            y.append(0)
        elif r["home_goals"] == r["away_goals"]:
            y.append(1)
        else:
            y.append(2)
    return np.array(y, dtype=int)


def simulate_value_betting(probs, raw_odds, y_true, edge_threshold=0.05, kelly_fraction=0.25, max_stake=0.05):
    """
    Simulates betting using a Fractional Kelly Criterion.
    - edge_threshold: Minimum EV to place a bet.
    - kelly_fraction: Multiplier for the Kelly fraction (e.g., 0.25 for Quarter Kelly) to reduce variance.
    - max_stake: Maximum allowed percentage of bankroll to risk on a single bet (e.g., 0.05 = 5%).
    """
    stats = {
        "Home (1)": {"count": 0, "wins": 0, "invested": 0.0, "return": 0.0, "odds_sum": 0.0},
        "Draw (X)": {"count": 0, "wins": 0, "invested": 0.0, "return": 0.0, "odds_sum": 0.0},
        "Away (2)": {"count": 0, "wins": 0, "invested": 0.0, "return": 0.0, "odds_sum": 0.0},
    }

    for i in range(len(probs)):
        p_h, p_d, p_a = probs[i]
        o_h, o_d, o_a = raw_odds[i]

        if not (np.isfinite(o_h) and np.isfinite(o_d) and np.isfinite(o_a)):
            continue

        # Add probabilities to the tuple so we can calculate Kelly later
        evs = [
            (p_h * o_h - 1, 0, o_h, "Home (1)", p_h),
            (p_d * o_d - 1, 1, o_d, "Draw (X)", p_d),
            (p_a * o_a - 1, 2, o_a, "Away (2)", p_a),
        ]

        # Find the best option based on EV
        best_ev, choice, odds_taken, label, prob_taken = max(evs, key=lambda x: x[0])

        if best_ev > edge_threshold:
            # 1. Calculate Kelly Stake: f* = EV / (odds - 1)
            b = odds_taken - 1.0
            f_star = best_ev / b if b > 0 else 0.0
            
            # 2. Apply Fractional Kelly and Cap it
            stake = f_star * kelly_fraction
            stake = min(stake, max_stake)
            
            # Minimum stake safety (e.g., don't place micro-bets less than 0.1%)
            if stake < 0.001:
                continue

            # 3. Record the bet
            stats[label]["count"] += 1
            stats[label]["invested"] += stake
            stats[label]["odds_sum"] += odds_taken

            if choice == y_true[i]:
                stats[label]["wins"] += 1
                stats[label]["return"] += stake * odds_taken

    print(f"\n{'Market Segment':<15} | {'Bets':<5} | {'Win%':<7} | {'ROI%':<8}")
    print("-" * 45)

    total_bets = 0
    total_wins = 0
    total_inv = 0.0
    total_ret = 0.0
    total_odds_sum = 0.0

    for label, s in stats.items():
        if s["count"] > 0:
            win_pc = (s["wins"] / s["count"]) * 100
            roi = ((s["return"] - s["invested"]) / s["invested"]) * 100
            print(f"{label:<15} | {s['count']:<5} | {win_pc:>6.1f}% | {roi:>7.2f}%")

            total_bets += s["count"]
            total_wins += s["wins"]
            total_inv += s["invested"]
            total_ret += s["return"]
            total_odds_sum += s["odds_sum"]

    final_profit = total_ret - total_inv
    final_roi = (final_profit / total_inv * 100) if total_inv > 0 else 0
    avg_odds = (total_odds_sum / total_bets) if total_bets > 0 else 0

    print("-" * 45)
    print(f"{'TOTAL':<15} | {total_bets:<5} | {'-':>7} | {final_roi:>7.2f}%")

    return total_bets, total_wins, final_profit, final_roi, avg_odds
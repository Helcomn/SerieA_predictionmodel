from __future__ import annotations

import pandas as pd

from src.config import FINAL_CONFIG


def _latest_rows(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "run_ts_utc" not in df.columns:
        return df
    return df[df["run_ts_utc"] == df["run_ts_utc"].max()].copy()


def main():
    model_rows = _latest_rows(FINAL_CONFIG.final_model_summary_file)
    ablation_rows = _latest_rows(FINAL_CONFIG.final_ablation_summary_file)
    robustness_rows = _latest_rows(FINAL_CONFIG.final_betting_robustness_file)
    league_selection_rows = _latest_rows(FINAL_CONFIG.final_league_model_selection_file)
    league_strategy_rows = _latest_rows(FINAL_CONFIG.final_league_strategy_file)

    print("=== FINAL MODEL SUMMARY ===")
    if model_rows.empty:
        print(f"No final model summary found at {FINAL_CONFIG.final_model_summary_file}")
    else:
        print(model_rows.sort_values("logloss").to_string(index=False))

    print("\n=== FINAL ABLATION SUMMARY ===")
    if ablation_rows.empty:
        print(f"No final ablation summary found at {FINAL_CONFIG.final_ablation_summary_file}")
    else:
        print(ablation_rows.sort_values(["model", "logloss"]).to_string(index=False))

    print("\n=== FINAL BETTING ROBUSTNESS ===")
    if robustness_rows.empty:
        print(f"No final betting robustness report found at {FINAL_CONFIG.final_betting_robustness_file}")
    else:
        overall = robustness_rows[robustness_rows["group_type"] == "all"].copy()
        by_league = robustness_rows[robustness_rows["group_type"] == "league"].copy()
        by_season = robustness_rows[robustness_rows["group_type"] == "season"].copy()

        if not overall.empty:
            print("\nOverall:")
            print(overall.sort_values("roi", ascending=False).to_string(index=False))
        if not by_league.empty:
            print("\nBy league:")
            print(by_league.sort_values(["model", "roi"], ascending=[True, False]).to_string(index=False))
        if not by_season.empty:
            print("\nBy season:")
            print(by_season.sort_values(["model", "group_value"]).to_string(index=False))

    print("\n=== FINAL LEAGUE-SPECIFIC STRATEGY ===")
    if league_selection_rows.empty:
        print(f"No league-specific model selection found at {FINAL_CONFIG.final_league_model_selection_file}")
    else:
        print("\nValidation-selected models:")
        print(league_selection_rows.sort_values("league").to_string(index=False))

    if league_strategy_rows.empty:
        print(f"No league-specific strategy report found at {FINAL_CONFIG.final_league_strategy_file}")
    else:
        print("\nFinal test strategy:")
        print(league_strategy_rows.sort_values(["group_type", "group_value"]).to_string(index=False))


if __name__ == "__main__":
    main()

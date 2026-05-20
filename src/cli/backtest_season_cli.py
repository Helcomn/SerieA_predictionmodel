from __future__ import annotations

import argparse

from src.config import ExperimentConfig
from src.trainer import run_training_pipeline


def season_window(season_start: int) -> tuple[str, str, str]:
    if season_start < 2014:
        raise ValueError("season must be 2014 or later so the pipeline has a full prior validation season")
    return (
        f"{season_start - 1}-07-01",
        f"{season_start}-07-01",
        f"{season_start + 1}-07-01",
    )


def build_backtest_config(
    season_start: int,
    *,
    force_refit: bool = False,
    force_retune: bool = False,
    full_report: bool = False,
) -> ExperimentConfig:
    train_cut, test_cut, test_end = season_window(season_start)
    return ExperimentConfig(
        experiment_name=f"season_backtest_{season_start}_{season_start + 1}",
        train_cut=train_cut,
        test_cut=test_cut,
        test_end=test_end,
        force_retune_leagues=force_retune,
        force_retune_meta=force_retune,
        force_refit_meta_model=force_refit or force_retune,
        force_retune_mlp=force_retune,
        force_refit_mlp_model=force_refit or force_retune,
        force_retune_blend=force_refit or force_retune,
        allow_partial_param_cache=True,
        generate_upcoming_picks=False,
        print_full_reports=full_report,
        print_parameter_impact=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a leakage-safe single-season betting backtest.",
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season start year. Example: --season 2024 evaluates 2024-2025.",
    )
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument(
        "--force-refit",
        action="store_true",
        help="Reuse cached season-specific tuning when available, but refit final models and retune the blend.",
    )
    refresh_group.add_argument(
        "--force-retune",
        action="store_true",
        help="Retune league, XGBoost, MLP, and blend settings for this season-specific backtest.",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Print the old detailed diagnostic tables. By default, detailed tables are written only to artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_backtest_config(
        args.season,
        force_refit=args.force_refit,
        force_retune=args.force_retune,
        full_report=args.full_report,
    )
    print("=== LEAKAGE-SAFE SEASON BACKTEST ===")
    print(f"Target season: {args.season}-{args.season + 1}")
    print(f"Fit history: dates before {config.train_cut}")
    print(f"Validation/meta window: [{config.train_cut}, {config.test_cut})")
    print(f"Test-only betting window: [{config.test_cut}, {config.test_end})")
    print("Match-state and rolling features are recomputed chronologically using only earlier dates.")
    run_training_pipeline(config)
    print("\nBacktest artifacts:")
    print(f"  Betting robustness: {config.final_betting_robustness_file}")
    print(f"  Bet selector: {config.final_bet_selector_file}")
    print(f"  League strategy: {config.final_league_strategy_file}")


if __name__ == "__main__":
    main()

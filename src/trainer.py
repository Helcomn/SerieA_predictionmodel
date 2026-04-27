from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from src.artifact_store import append_rows_to_csv, load_json_if_exists, load_pickle_if_exists, save_json, save_manifest, save_pickle
from src.betting_robustness import write_betting_robustness_report, write_league_specific_strategy_report
from src.calibration import temperature_scale_probs
from src.config import DEFAULT_CONFIG, ExperimentConfig
from src.data_loader import load_league_data
from src.evaluation import (
    simulate_value_betting,
    print_alignment_audit,
    print_strategy_comparison,
    print_market_dependency_audit,
    print_profit_profile_audit,
)
from src.feature_builder import (
    BASE_NON_MARKET_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LOCAL_STATS_FEATURE_COLUMNS,
    MARKET_CONTEXT_FEATURE_COLUMNS,
    MLP_DEFAULT_COLS,
    MLP_DEFAULT_FEATURE_COLUMNS,
    MLP_NO_REST_COLS,
    NEW_DATA_FEATURE_COLUMNS,
    NEW_LOCAL_FEATURE_COLUMNS,
    ODDS_MOVEMENT_FEATURE_COLUMNS,
    UNDERSTAT_XG_FEATURE_COLUMNS,
    build_meta_features,
    ensure_market_probs,
    feature_indices,
    time_split_val,
    market_probs_from_odds_row,
)
from src.models.base import tune_league_params
from src.models.meta import (
    blend_probabilities,
    fit_xgb_model,
    make_logreg_pipeline,
    make_mlp_pipeline,
    probs_from_meta_features,
    tune_feature_subset,
    tune_blend_weights,
    tune_logreg_hyperparams,
    tune_mlp_feature_subset,
    tune_mlp_hyperparams,
    tune_xgb_hyperparams,
)
from src.reporting import print_per_league_test_metrics
from src.reporting_ext import print_confusion, print_prob_report
from src.services.upcoming import generate_upcoming_matchday_picks
from src.state_builder import streaming_block_probs_home_away


PIPELINE_VERSION = 7


def _cached_artifacts_are_compatible(config: ExperimentConfig) -> bool:
    manifest = load_json_if_exists(config.manifest_file)
    if manifest is None:
        print("Cached artifact manifest missing; retuning/refitting artifacts.")
        return False
    if manifest.get("pipeline_version") != PIPELINE_VERSION:
        print("Cached artifacts were produced by an older pipeline; retuning/refitting artifacts.")
        return False

    manifest_config = manifest.get("config", {})
    expected_config = config.as_manifest()
    for key in ("experiment_name", "train_cut", "test_cut", "leagues"):
        if manifest_config.get(key) != expected_config.get(key):
            print(f"Cached artifact config mismatch for {key}; retuning/refitting artifacts.")
            return False

    if manifest.get("feature_columns") != FEATURE_COLUMNS:
        print("Cached artifact feature schema mismatch; retuning/refitting artifacts.")
        return False
    mlp_feature_columns = manifest.get("mlp_feature_columns")
    if not mlp_feature_columns or any(col not in FEATURE_COLUMNS for col in mlp_feature_columns):
        print("Cached artifact MLP feature schema invalid; retuning/refitting artifacts.")
        return False
    return True


def _model_summary_row(name: str, probs: np.ndarray, y_true: np.ndarray, raw_odds: np.ndarray, *, edge_threshold: float):
    bets, wins, profit, roi, avg_odds = simulate_value_betting(
        probs,
        raw_odds,
        y_true,
        edge_threshold=edge_threshold,
        match_info=None,
        verbose=False,
    )
    accuracy = float((np.argmax(probs, axis=1) == y_true).mean())
    hit_rate = (wins / bets * 100.0) if bets > 0 else 0.0
    return {
        "name": name,
        "logloss": float(log_loss(y_true, probs)),
        "accuracy": accuracy,
        "bets": int(bets),
        "hit_rate": float(hit_rate),
        "roi": float(roi),
        "profit": float(profit),
        "avg_odds": float(avg_odds),
    }


def _print_model_selection_summary(rows: list[dict]):
    print("\n" + "=" * 70)
    print("=== MODEL SELECTION SUMMARY ===")
    print("=" * 70)

    print(
        f"{'Model':<12} | {'LogLoss':>8} | {'Acc%':>6} | {'Bets':>5} | {'Hit%':>6} | {'ROI%':>7} | {'Profit':>8}"
    )
    print("-" * 70)
    for row in sorted(rows, key=lambda r: (r["logloss"], -r["roi"])):
        print(
            f"{row['name']:<12} | {row['logloss']:>8.4f} | {row['accuracy'] * 100:>6.2f} | {row['bets']:>5} | "
            f"{row['hit_rate']:>6.2f} | {row['roi']:>7.2f} | {row['profit']:>8.3f}"
        )

    best_logloss = min(rows, key=lambda r: r["logloss"])
    best_roi = max(rows, key=lambda r: r["roi"])

    print("\nWinner by LogLoss:", best_logloss["name"], f"({best_logloss['logloss']:.4f})")
    print("Winner by Betting ROI:", best_roi["name"], f"({best_roi['roi']:.2f}%)")
    return best_logloss, best_roi


def _select_recommended_betting_model(rows: list[dict], probability_winner: dict):
    eligible_names = {"market", "logreg", "meta", "ensemble"}
    eligible = [row for row in rows if row["name"] in eligible_names and np.isfinite(row["logloss"])]
    if not eligible:
        return probability_winner, "fallback_to_probability_winner"

    best_roi = max(eligible, key=lambda r: r["roi"])
    if best_roi["roi"] <= 0:
        return probability_winner, "no_positive_roi_among_stable_candidates"

    logloss_gap = best_roi["logloss"] - probability_winner["logloss"]
    if logloss_gap > 0.01:
        return probability_winner, "positive_roi_candidate_failed_logloss_guard"
    return best_roi, "positive_roi_stable_candidate"


def _write_final_summary(config: ExperimentConfig, selection_rows: list[dict], ablation_rows: list[dict], run_ts: str):
    final_rows = []
    for row in selection_rows:
        final_rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "model": row["name"],
            "logloss": round(row["logloss"], 6),
            "accuracy": round(row["accuracy"], 6),
            "bets": row["bets"],
            "hit_rate": round(row["hit_rate"], 4),
            "roi": round(row["roi"], 4),
            "profit": round(row["profit"], 4),
            "avg_odds": round(row["avg_odds"], 4),
        })
    append_rows_to_csv(config.final_model_summary_file, final_rows)

    thesis_feature_sets = {
        "market_only",
        "market_plus_context",
        "market_plus_understat_xg",
        "market_context_plus_understat_xg",
        "core_18",
        "no_market",
        "no_understat_xg",
    }
    final_ablation_rows = []
    for row in ablation_rows:
        if row["feature_set"] in thesis_feature_sets:
            final_ablation_rows.append({
                "run_ts_utc": run_ts,
                "experiment_name": config.experiment_name,
                "model": row["model"],
                "feature_set": row["feature_set"],
                "logloss": round(row["logloss"], 6),
                "accuracy": round(row["accuracy"], 6),
            })
    append_rows_to_csv(config.final_ablation_summary_file, final_ablation_rows)


def _print_feature_ablation_summary(rows: list[dict]):
    print("\n" + "=" * 70)
    print("=== FEATURE GROUP ABLATION ===")
    print("=" * 70)
    print(f"{'Model':<8} | {'Feature Set':<18} | {'LogLoss':>8} | {'Accuracy%':>9}")
    print("-" * 70)
    for row in sorted(rows, key=lambda r: (r["model"], r["logloss"])):
        print(
            f"{row['model']:<8} | {row['feature_set']:<18} | {row['logloss']:>8.4f} | {row['accuracy'] * 100:>9.2f}"
        )


def _ablation_row(model_name: str, feature_set: str, probs: np.ndarray, y_true: np.ndarray):
    return {
        "model": model_name,
        "feature_set": feature_set,
        "logloss": float(log_loss(y_true, probs)),
        "accuracy": float((np.argmax(probs, axis=1) == y_true).mean()),
    }


def run_training_pipeline(config: ExperimentConfig = DEFAULT_CONFIG):
    compatible_cache = config.use_cached_artifacts and _cached_artifacts_are_compatible(config)
    cached_params = load_json_if_exists(config.params_file) if config.use_cached_artifacts and not config.force_retune_leagues else None
    cached_meta = load_json_if_exists(config.meta_file) if compatible_cache and not config.force_retune_meta else None
    cached_mlp = load_json_if_exists(config.mlp_meta_file) if compatible_cache and not config.force_retune_mlp else None
    cached_blend = load_json_if_exists(config.blend_file) if compatible_cache and not config.force_retune_blend else None
    league_best_params = {} if cached_params is None else cached_params
    all_X_early, all_y_early = [], []
    all_X_late, all_y_late = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []
    all_v_probs_model, all_v_mkt_fixed, all_v_raw_odds, all_v_info = [], [], [], []
    all_t_probs_model, all_t_mkt_fixed, all_t_raw_odds, all_t_info = [], [], [], []
    per_league_test = {}

    for league in config.leagues:
        print("\n" + "=" * 50)
        print(f"=== Processing Data: {league.upper()} ===")
        print("=" * 50)
        df_all = load_league_data(league).sort_values("date").reset_index(drop=True)
        df = df_all[df_all["is_played"] == True].copy().reset_index(drop=True)
        if df.empty:
            continue

        train_fit = df[df["date"] < config.train_cut].copy()
        val = df[(df["date"] >= config.train_cut) & (df["date"] < config.test_cut)].copy()
        test = df[df["date"] >= config.test_cut].copy()
        print(f"Train_fit: {len(train_fit)}, Validation: {len(val)}, Test: {len(test)}")
        if len(train_fit) == 0 or len(val) == 0 or len(test) == 0:
            continue

        base_train, base_param_val = time_split_val(train_fit)

        if league in league_best_params and not config.force_retune_leagues:
            params = league_best_params[league]
            print("\n--- Using cached params ---")
            print(f"K={params['K']}, ha={params['ha']}, beta={params['beta']}, decay={params['decay']}, rho={params['rho']}, T={params['T']}")
        else:
            params = tune_league_params(
                base_train,
                base_param_val,
                df[df["date"] < config.train_cut].copy(),
            )
            league_best_params[league] = params

        val_early, val_late = time_split_val(val)
        ve_probs_raw, ve_y, ve_mkt, ve_aux, _ = streaming_block_probs_home_away(val_early, df, params["beta"], params["rho"], params["decay"], params["K"], params["ha"])
        vl_probs_raw, vl_y, vl_mkt, vl_aux, _ = streaming_block_probs_home_away(val_late, df, params["beta"], params["rho"], params["decay"], params["K"], params["ha"])
        ve_probs_model = temperature_scale_probs(ve_probs_raw, params["T"])
        vl_probs_model = temperature_scale_probs(vl_probs_raw, params["T"])

        all_X_early.extend(build_meta_features(ve_probs_model, ve_mkt, ve_aux))
        all_y_early.extend(ve_y)
        all_X_late.extend(build_meta_features(vl_probs_model, vl_mkt, vl_aux))
        all_y_late.extend(vl_y)

        v_probs_raw, v_y, v_mkt, v_aux, v_raw_odds = streaming_block_probs_home_away(val, df, params["beta"], params["rho"], params["decay"], params["K"], params["ha"])
        v_probs_model = temperature_scale_probs(v_probs_raw, params["T"])
        v_mkt_fixed = ensure_market_probs(v_probs_model, v_mkt)
        all_X_val.extend(build_meta_features(v_probs_model, v_mkt_fixed, v_aux))
        all_y_val.extend(v_y)
        all_v_probs_model.extend(v_probs_model)
        all_v_mkt_fixed.extend(v_mkt_fixed)
        all_v_raw_odds.extend(v_raw_odds)
        val_info = val[["date", "home_team", "away_team"]].to_dict("records")
        for row in val_info:
            row["league"] = league
        all_v_info.extend(val_info)

        t_probs_raw, t_y, t_mkt, t_aux, t_raw_odds = streaming_block_probs_home_away(test, df, params["beta"], params["rho"], params["decay"], params["K"], params["ha"])
        t_probs_model = temperature_scale_probs(t_probs_raw, params["T"])
        t_mkt_fixed = ensure_market_probs(t_probs_model, t_mkt)

        all_X_test.extend(build_meta_features(t_probs_model, t_mkt_fixed, t_aux))
        all_y_test.extend(t_y)
        all_t_probs_model.extend(t_probs_model)
        all_t_mkt_fixed.extend(t_mkt_fixed)
        all_t_raw_odds.extend(t_raw_odds)
        test_info = test[["date", "home_team", "away_team"]].to_dict("records")
        for row in test_info:
            row["league"] = league
        all_t_info.extend(test_info)
        per_league_test[league] = {"y": np.array(t_y, dtype=int), "p_model": np.array(t_probs_model, dtype=float), "p_mkt": np.array(t_mkt_fixed, dtype=float)}

    if cached_params is None or config.force_retune_leagues:
        save_json(config.params_file, league_best_params)
        print(f"\nSaved tuned league params to: {config.params_file}")

    X_early_arr, y_early_arr = np.array(all_X_early), np.array(all_y_early)
    X_late_arr, y_late_arr = np.array(all_X_late), np.array(all_y_late)
    X_val_arr, y_val_arr = np.array(all_X_val), np.array(all_y_val)
    X_test_arr, y_test_arr = np.array(all_X_test), np.array(all_y_test)
    v_probs_model_arr, v_mkt_fixed_arr = np.array(all_v_probs_model), np.array(all_v_mkt_fixed)
    t_probs_model_arr, t_mkt_fixed_arr = np.array(all_t_probs_model), np.array(all_t_mkt_fixed)
    val_raw_odds_arr = np.array(all_v_raw_odds)
    raw_odds_arr = np.array(all_t_raw_odds)

    threshold = 0.05

    all_cols = list(range(len(FEATURE_COLUMNS)))
    core_feature_names = FEATURE_COLUMNS[:18]
    CORE_COLS = feature_indices(core_feature_names)
    LOCAL_STATS_COLS = feature_indices(LOCAL_STATS_FEATURE_COLUMNS)
    MARKET_CONTEXT_COLS = feature_indices(MARKET_CONTEXT_FEATURE_COLUMNS)
    NEW_LOCAL_COLS = feature_indices(NEW_LOCAL_FEATURE_COLUMNS)
    CORE_PLUS_ODDS_COLS = feature_indices(core_feature_names + MARKET_CONTEXT_FEATURE_COLUMNS)
    CORE_PLUS_STATS_COLS = feature_indices(core_feature_names + LOCAL_STATS_FEATURE_COLUMNS)
    CORE_PLUS_NEW_LOCAL_COLS = feature_indices(core_feature_names + NEW_LOCAL_FEATURE_COLUMNS)
    CORE_PLUS_UNDERSTAT_XG_COLS = feature_indices(core_feature_names + UNDERSTAT_XG_FEATURE_COLUMNS)
    MARKET_PLUS_UNDERSTAT_XG_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *UNDERSTAT_XG_FEATURE_COLUMNS,
    ])
    MARKET_CONTEXT_PLUS_UNDERSTAT_XG_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *MARKET_CONTEXT_FEATURE_COLUMNS,
        *UNDERSTAT_XG_FEATURE_COLUMNS,
    ])
    NO_NEW_LOCAL_FEATURES_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in NEW_LOCAL_FEATURE_COLUMNS]
    NO_NEW_DATA_FEATURES_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in NEW_DATA_FEATURE_COLUMNS]
    NO_NEW_LOCAL_STATS_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in LOCAL_STATS_FEATURE_COLUMNS]
    NO_ODDS_MOVEMENT_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in MARKET_CONTEXT_FEATURE_COLUMNS]
    NO_UNDERSTAT_XG_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in UNDERSTAT_XG_FEATURE_COLUMNS]
    market_feature_names = {
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        "market_move_home",
        "market_move_draw",
        "market_move_away",
        "ou25_over_prob",
        "ah_line",
    }
    NO_MARKET_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in market_feature_names]
    NO_ELO_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col != "elo_diff"]
    NO_XG_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in {"total_xg", "xg_diff"}]
    NO_MOMENTUM_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if not col.startswith("mom_")]
    NO_REST_COLS = MLP_NO_REST_COLS
    NO_FORM_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if not col.startswith("form_")]
    MODEL_ONLY_COLS = [0, 1, 2]
    MARKET_ONLY_COLS = [3, 4, 5]
    MARKET_PLUS_CONTEXT_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *MARKET_CONTEXT_FEATURE_COLUMNS,
    ])
    learned_model_feature_sets = {
        "all_features": all_cols,
        "core_18": CORE_COLS,
        "market_only": MARKET_ONLY_COLS,
        "market_plus_context": MARKET_PLUS_CONTEXT_COLS,
        "core_plus_odds": CORE_PLUS_ODDS_COLS,
        "core_plus_stats": CORE_PLUS_STATS_COLS,
        "core_plus_new_local": CORE_PLUS_NEW_LOCAL_COLS,
        "core_plus_understat_xg": CORE_PLUS_UNDERSTAT_XG_COLS,
        "market_plus_understat_xg": MARKET_PLUS_UNDERSTAT_XG_COLS,
        "market_context_plus_understat_xg": MARKET_CONTEXT_PLUS_UNDERSTAT_XG_COLS,
        "no_new_local_stats": NO_NEW_LOCAL_STATS_COLS,
        "no_odds_context": NO_ODDS_MOVEMENT_COLS,
        "no_understat_xg": NO_UNDERSTAT_XG_COLS,
    }

    X_early_nm = X_early_arr[:, NO_MARKET_COLS]
    X_late_nm = X_late_arr[:, NO_MARKET_COLS]
    X_val_nm = X_val_arr[:, NO_MARKET_COLS]
    X_test_nm = X_test_arr[:, NO_MARKET_COLS]

    recomputed_market = np.array(
        [market_probs_from_odds_row(o[0], o[1], o[2]) for o in raw_odds_arr],
        dtype=float,
    )

    mask = np.isfinite(recomputed_market).all(axis=1)
    if np.any(mask):
        max_diff = np.max(np.abs(recomputed_market[mask] - t_mkt_fixed_arr[mask]))
        print("SANITY max market diff:", max_diff)

        ev_check = recomputed_market[mask] * raw_odds_arr[mask] - 1.0
        print("SANITY max raw market EV:", np.max(ev_check))
    else:
        print("SANITY market odds check skipped: no valid raw market odds in test set.")

    print("\n" + "=" * 50)
    print("=== META-MODEL Evaluation (XGBoost) ===")
    print("=" * 50)

    if cached_meta is not None and not config.force_retune_meta:
        best_meta_cfg = cached_meta
        print("Using cached XGBoost hyperparameters...")
    else:
        best_meta_cfg = tune_xgb_hyperparams(X_early_arr, y_early_arr, X_late_arr, y_late_arr)
        xgb_subset, xgb_subset_rows = tune_feature_subset(
            lambda: XGBClassifier(
                n_estimators=int(best_meta_cfg["n_estimators"]),
                learning_rate=float(best_meta_cfg["learning_rate"]),
                max_depth=int(best_meta_cfg["max_depth"]),
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=config.random_state,
                n_jobs=-1,
            ),
            X_early_arr,
            y_early_arr,
            X_late_arr,
            y_late_arr,
            learned_model_feature_sets,
        )
        best_meta_cfg["feature_set"] = xgb_subset["name"]
        best_meta_cfg["feature_columns"] = [FEATURE_COLUMNS[i] for i in xgb_subset["cols"]]
        best_meta_cfg["late_val_logloss"] = float(xgb_subset["late_val_logloss"])
        best_meta_cfg["feature_subset_scores"] = [
            {
                "name": row["name"],
                "late_val_logloss": row["late_val_logloss"],
                "feature_columns": [FEATURE_COLUMNS[i] for i in row["cols"]],
            }
            for row in xgb_subset_rows
        ]
        save_json(config.meta_file, best_meta_cfg)
        print(f"Saved XGBoost hyperparams to: {config.meta_file}")
    print(f"Best XGBoost Config -> LR: {best_meta_cfg['learning_rate']}, Depth: {best_meta_cfg['max_depth']}, Trees: {best_meta_cfg['n_estimators']}")
    print(f"Late VAL LogLoss: {round(best_meta_cfg['late_val_logloss'], 4)}")
    xgb_feature_columns = best_meta_cfg.get("feature_columns", FEATURE_COLUMNS)
    xgb_cols = feature_indices(xgb_feature_columns)
    print(f"XGBoost feature set: {best_meta_cfg.get('feature_set', 'all_features')} ({len(xgb_cols)} features)")

    can_load_meta_model = compatible_cache and config.model_file.exists() and not config.force_retune_meta and not config.force_refit_meta_model
    if can_load_meta_model:
        print(f"Loading trained XGBoost meta-model from: {config.model_file}")
        meta_final = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=config.random_state, n_jobs=-1)
        meta_final.load_model(str(config.model_file))
    else:
        print("Fitting final XGBoost meta-model...")
        meta_final = fit_xgb_model(X_val_arr[:, xgb_cols], y_val_arr, best_meta_cfg)
        config.model_file.parent.mkdir(parents=True, exist_ok=True)
        meta_final.save_model(str(config.model_file))
        print(f"Saved trained XGBoost meta-model to: {config.model_file}")

    mlp_candidate_feature_sets = {
        "default": MLP_DEFAULT_COLS,
        "default_plus_odds_context": sorted(set(MLP_DEFAULT_COLS + MARKET_CONTEXT_COLS)),
        "default_plus_rolling_stats": sorted(set(MLP_DEFAULT_COLS + LOCAL_STATS_COLS)),
        "default_plus_new_local": sorted(set(MLP_DEFAULT_COLS + NEW_LOCAL_COLS)),
        "default_plus_understat_xg": sorted(set(MLP_DEFAULT_COLS + feature_indices(UNDERSTAT_XG_FEATURE_COLUMNS))),
        "core_18": CORE_COLS,
        "core_plus_odds_context": CORE_PLUS_ODDS_COLS,
        "core_plus_rolling_stats": CORE_PLUS_STATS_COLS,
        "core_plus_understat_xg": CORE_PLUS_UNDERSTAT_XG_COLS,
        "all_features": all_cols,
    }

    if cached_mlp is not None and not config.force_retune_mlp:
        best_mlp_cfg = cached_mlp
        print("Using cached MLP hyperparameters...")
    else:
        base_mlp_cfg = tune_mlp_hyperparams(
            X_early_arr[:, MLP_DEFAULT_COLS],
            y_early_arr,
            X_late_arr[:, MLP_DEFAULT_COLS],
            y_late_arr,
        )
        best_subset, subset_rows = tune_mlp_feature_subset(
            X_early_arr,
            y_early_arr,
            X_late_arr,
            y_late_arr,
            base_mlp_cfg,
            mlp_candidate_feature_sets,
        )
        best_mlp_cfg = dict(base_mlp_cfg)
        best_mlp_cfg["feature_set"] = best_subset["name"]
        best_mlp_cfg["feature_columns"] = [FEATURE_COLUMNS[i] for i in best_subset["cols"]]
        best_mlp_cfg["late_val_logloss"] = float(best_subset["late_val_logloss"])
        best_mlp_cfg["temperature"] = float(best_subset["temperature"])
        best_mlp_cfg["feature_subset_scores"] = [
            {
                "name": row["name"],
                "late_val_logloss": row["late_val_logloss"],
                "feature_columns": [FEATURE_COLUMNS[i] for i in row["cols"]],
            }
            for row in subset_rows
        ]
        save_json(config.mlp_meta_file, best_mlp_cfg)
        print(f"Saved MLP hyperparams to: {config.mlp_meta_file}")

    mlp_feature_columns = best_mlp_cfg.get("feature_columns", MLP_DEFAULT_FEATURE_COLUMNS)
    mlp_cols = feature_indices(mlp_feature_columns)
    X_val_mlp = X_val_arr[:, mlp_cols]
    X_test_mlp = X_test_arr[:, mlp_cols]
    X_early_mlp = X_early_arr[:, mlp_cols]
    X_late_mlp = X_late_arr[:, mlp_cols]

    can_load_mlp_model = compatible_cache and config.mlp_model_file.exists() and not config.force_retune_mlp and not config.force_refit_mlp_model
    if can_load_mlp_model:
        print(f"Loading trained MLP model from: {config.mlp_model_file}")
        mlp_model = load_pickle_if_exists(config.mlp_model_file)
    else:
        print("Fitting final MLP model...")
        mlp_model = make_mlp_pipeline(best_mlp_cfg)
        mlp_model.fit(X_val_mlp, y_val_arr)
        save_pickle(config.mlp_model_file, mlp_model)
        print(f"Saved trained MLP model to: {config.mlp_model_file}")

    logreg_cfg = tune_logreg_hyperparams(
        X_early_arr[:, MARKET_PLUS_CONTEXT_COLS],
        y_early_arr,
        X_late_arr[:, MARKET_PLUS_CONTEXT_COLS],
        y_late_arr,
    )
    logreg_subset, logreg_subset_rows = tune_feature_subset(
        lambda: make_logreg_pipeline(logreg_cfg["C"]),
        X_early_arr,
        y_early_arr,
        X_late_arr,
        y_late_arr,
        learned_model_feature_sets,
        temperature_scale=True,
    )
    logreg_feature_columns = [FEATURE_COLUMNS[i] for i in logreg_subset["cols"]]
    logreg_cols = feature_indices(logreg_feature_columns)
    logreg_cfg["feature_set"] = logreg_subset["name"]
    logreg_cfg["feature_columns"] = logreg_feature_columns
    logreg_cfg["late_val_logloss"] = float(logreg_subset["late_val_logloss"])
    logreg_cfg["temperature"] = float(logreg_subset["temperature"])
    logreg_cfg["feature_subset_scores"] = [
        {
            "name": row["name"],
            "late_val_logloss": row["late_val_logloss"],
            "feature_columns": [FEATURE_COLUMNS[i] for i in row["cols"]],
        }
        for row in logreg_subset_rows
    ]
    print(f"LogReg feature set: {logreg_cfg['feature_set']} ({len(logreg_cols)} features)")
    logreg_final = make_logreg_pipeline(logreg_cfg["C"])
    logreg_final.fit(X_val_arr[:, logreg_cols], y_val_arr)

    xgb_late_model = fit_xgb_model(X_early_arr[:, xgb_cols], y_early_arr, best_meta_cfg)
    late_probs_xgb = xgb_late_model.predict_proba(X_late_arr[:, xgb_cols])
    mlp_late_model = make_mlp_pipeline(best_mlp_cfg)
    mlp_late_model.fit(X_early_mlp, y_early_arr)
    late_probs_mlp_raw = mlp_late_model.predict_proba(X_late_mlp)
    late_probs_mlp = temperature_scale_probs(late_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    late_probs_base = probs_from_meta_features(X_late_arr, 0)
    late_probs_mkt = probs_from_meta_features(X_late_arr, 3)
    late_probs_logreg_raw = logreg_final.predict_proba(X_late_arr[:, logreg_cols])
    late_probs_logreg = temperature_scale_probs(late_probs_logreg_raw, float(logreg_cfg["temperature"]))
    late_market_ll = float(log_loss(y_late_arr, late_probs_mkt))
    late_mlp_ll = float(log_loss(y_late_arr, late_probs_mlp))
    mlp_allowed_in_blend = late_mlp_ll < late_market_ll - 0.002
    if not mlp_allowed_in_blend:
        print(
            f"MLP excluded from blend: late_val_logloss={late_mlp_ll:.4f}, "
            f"market={late_market_ll:.4f}"
        )

    if cached_blend is not None and not config.force_retune_blend:
        blend_cfg = cached_blend
        print("Using cached blend weights...")
    else:
        blend_cfg = tune_blend_weights(
            y_late_arr,
            late_probs_base,
            late_probs_mkt,
            late_probs_xgb,
            late_probs_mlp if mlp_allowed_in_blend else None,
            step=0.05,
        )
        blend_cfg["mlp_allowed"] = bool(mlp_allowed_in_blend)
        blend_cfg["mlp_gate"] = {"late_mlp_logloss": late_mlp_ll, "late_market_logloss": late_market_ll, "margin_required": 0.002}
        save_json(config.blend_file, blend_cfg)
        print(f"Saved blend weights to: {config.blend_file}")
    print(f"Blend weights: {blend_cfg['weights']}")
    print(f"Late VAL LogLoss: {round(blend_cfg['late_val_logloss'], 4)}")

    v_probs_meta = meta_final.predict_proba(X_val_arr[:, xgb_cols])
    v_probs_mlp_raw = mlp_model.predict_proba(X_val_mlp)
    v_probs_mlp = temperature_scale_probs(v_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    v_probs_logreg_raw = logreg_final.predict_proba(X_val_arr[:, logreg_cols])
    v_probs_logreg = temperature_scale_probs(v_probs_logreg_raw, float(logreg_cfg["temperature"]))

    t_probs_meta = meta_final.predict_proba(X_test_arr[:, xgb_cols])
    t_probs_mlp_raw = mlp_model.predict_proba(X_test_mlp)
    t_probs_mlp = temperature_scale_probs(t_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    t_probs_logreg_raw = logreg_final.predict_proba(X_test_arr[:, logreg_cols])
    t_probs_logreg = temperature_scale_probs(t_probs_logreg_raw, float(logreg_cfg["temperature"]))
    t_probs_blend = blend_probabilities(
        blend_cfg["weights"],
        {
            "base": t_probs_model_arr,
            "market": t_mkt_fixed_arr,
            "xgb": t_probs_meta,
            "mlp": t_probs_mlp if blend_cfg.get("mlp_allowed", mlp_allowed_in_blend) else None,
        },
    )

    print("\n" + "=" * 50)
    print("=== QUICK NO-MARKET ABLATION ===")
    print("=" * 50)

    # Reuse current best hyperparams, but retrain on no-market features only
    meta_nm = fit_xgb_model(X_val_nm, y_val_arr, best_meta_cfg)
    t_probs_meta_nm = meta_nm.predict_proba(X_test_nm)

    mlp_nm = make_mlp_pipeline(best_mlp_cfg)
    mlp_nm.fit(X_val_nm, y_val_arr)
    t_probs_mlp_nm_raw = mlp_nm.predict_proba(X_test_nm)
    t_probs_mlp_nm = temperature_scale_probs(t_probs_mlp_nm_raw, float(best_mlp_cfg["temperature"]))

    print_prob_report("META NO-MARKET", t_probs_meta_nm, y_test_arr)
    print_prob_report("MLP NO-MARKET", t_probs_mlp_nm, y_test_arr)

    print("\n--- Betting simulation: no-market ablation ---")
    bets_nm_xgb, wins_nm_xgb, profit_nm_xgb, roi_nm_xgb, avg_odds_nm_xgb = simulate_value_betting(
        t_probs_meta_nm, raw_odds_arr, y_test_arr, edge_threshold=threshold, match_info=None
    )
    print(f"META NO-MARKET ROI: {round(roi_nm_xgb, 2)}% | Bets: {bets_nm_xgb}")

    bets_nm_mlp, wins_nm_mlp, profit_nm_mlp, roi_nm_mlp, avg_odds_nm_mlp = simulate_value_betting(
        t_probs_mlp_nm, raw_odds_arr, y_test_arr, edge_threshold=threshold, match_info=None
    )
    print(f"MLP NO-MARKET ROI: {round(roi_nm_mlp, 2)}% | Bets: {bets_nm_mlp}")

    print_prob_report("BASE (Model only, calibrated)", t_probs_model_arr, y_test_arr)
    print_prob_report("MARKET (odds implied)", t_mkt_fixed_arr, y_test_arr)
    print_prob_report("META (Market + Model)", t_probs_meta, y_test_arr)
    print_prob_report("LOGREG (regularized baseline)", t_probs_logreg, y_test_arr)
    print_prob_report("DEEP LEARNING (MLP, calibrated)", t_probs_mlp, y_test_arr)
    print_prob_report("ENSEMBLE (XGB + MLP + Market + Base)", t_probs_blend, y_test_arr)
    print_confusion("ENSEMBLE", t_probs_blend, y_test_arr)
    print_per_league_test_metrics(list(config.leagues), per_league_test, t_probs_meta, t_probs_mlp, t_probs_blend)

    ablation_specs = [
        ("all_features", all_cols),
        ("core_18", CORE_COLS),
        ("no_new_data_features", NO_NEW_DATA_FEATURES_COLS),
        ("no_new_local_features", NO_NEW_LOCAL_FEATURES_COLS),
        ("no_new_local_stats", NO_NEW_LOCAL_STATS_COLS),
        ("no_understat_xg", NO_UNDERSTAT_XG_COLS),
        ("no_odds_context", NO_ODDS_MOVEMENT_COLS),
        ("core_plus_odds", CORE_PLUS_ODDS_COLS),
        ("core_plus_stats", CORE_PLUS_STATS_COLS),
        ("core_plus_new_local", CORE_PLUS_NEW_LOCAL_COLS),
        ("core_plus_understat_xg", CORE_PLUS_UNDERSTAT_XG_COLS),
        ("market_plus_understat_xg", MARKET_PLUS_UNDERSTAT_XG_COLS),
        ("market_context_plus_understat_xg", MARKET_CONTEXT_PLUS_UNDERSTAT_XG_COLS),
        ("no_market", NO_MARKET_COLS),
        ("no_elo", NO_ELO_COLS),
        ("no_xg", NO_XG_COLS),
        ("no_momentum", NO_MOMENTUM_COLS),
        ("no_rest", NO_REST_COLS),
        ("no_form", NO_FORM_COLS),
        ("model_only", MODEL_ONLY_COLS),
        ("market_only", MARKET_ONLY_COLS),
        ("market_plus_context", MARKET_PLUS_CONTEXT_COLS),
    ]
    ablation_rows = []
    for feature_set_name, cols in ablation_specs:
        X_val_slice = X_val_arr[:, cols]
        X_test_slice = X_test_arr[:, cols]

        xgb_ablation = fit_xgb_model(X_val_slice, y_val_arr, best_meta_cfg)
        xgb_probs = xgb_ablation.predict_proba(X_test_slice)
        ablation_rows.append(_ablation_row("xgb", feature_set_name, xgb_probs, y_test_arr))

        mlp_ablation = make_mlp_pipeline(best_mlp_cfg)
        mlp_ablation.fit(X_val_slice, y_val_arr)
        mlp_probs_raw = mlp_ablation.predict_proba(X_test_slice)
        mlp_probs = temperature_scale_probs(mlp_probs_raw, float(best_mlp_cfg["temperature"]))
        ablation_rows.append(_ablation_row("mlp", feature_set_name, mlp_probs, y_test_arr))
    _print_feature_ablation_summary(ablation_rows)

    selection_rows = [
        _model_summary_row("base", t_probs_model_arr, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("market", t_mkt_fixed_arr, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("meta", t_probs_meta, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("logreg", t_probs_logreg, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("mlp", t_probs_mlp, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("ensemble", t_probs_blend, y_test_arr, raw_odds_arr, edge_threshold=threshold),
    ]
    best_logloss, best_roi = _print_model_selection_summary(selection_rows)
    recommended_betting, betting_reason = _select_recommended_betting_model(selection_rows, best_logloss)
    print("Recommended betting model:", recommended_betting["name"], f"({betting_reason})")

    run_ts = datetime.now(UTC).isoformat()
    result_rows = []
    for row in selection_rows:
        result_rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "train_cut": config.train_cut,
            "test_cut": config.test_cut,
            "model": row["name"],
            "logloss": round(row["logloss"], 6),
            "accuracy": round(row["accuracy"], 6),
            "bets": row["bets"],
            "hit_rate": round(row["hit_rate"], 4),
            "roi": round(row["roi"], 4),
            "profit": round(row["profit"], 4),
            "avg_odds": round(row["avg_odds"], 4),
            "winner_logloss": int(row["name"] == min(selection_rows, key=lambda r: r["logloss"])["name"]),
            "winner_roi": int(row["name"] == max(selection_rows, key=lambda r: r["roi"])["name"]),
        })
    append_rows_to_csv(config.results_csv_file, result_rows)

    ablation_csv_rows = []
    for row in ablation_rows:
        ablation_csv_rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "model": row["model"],
            "feature_set": row["feature_set"],
            "logloss": round(row["logloss"], 6),
            "accuracy": round(row["accuracy"], 6),
        })
    append_rows_to_csv(config.ablations_csv_file, ablation_csv_rows)
    _write_final_summary(config, selection_rows, ablation_rows, run_ts)
    write_betting_robustness_report(
        config,
        run_ts,
        {
            "base": t_probs_model_arr,
            "market": t_mkt_fixed_arr,
            "meta": t_probs_meta,
            "logreg": t_probs_logreg,
            "mlp": t_probs_mlp,
            "ensemble": t_probs_blend,
        },
        raw_odds_arr,
        y_test_arr,
        all_t_info,
        edge_threshold=threshold,
    )
    write_league_specific_strategy_report(
        config,
        run_ts,
        {
            "base": v_probs_model_arr,
            "market": v_mkt_fixed_arr,
            "meta": v_probs_meta,
            "logreg": v_probs_logreg,
            "mlp": v_probs_mlp,
        },
        val_raw_odds_arr,
        y_val_arr,
        all_v_info,
        {
            "base": t_probs_model_arr,
            "market": t_mkt_fixed_arr,
            "meta": t_probs_meta,
            "logreg": t_probs_logreg,
            "mlp": t_probs_mlp,
        },
        raw_odds_arr,
        y_test_arr,
        all_t_info,
        edge_threshold=threshold,
    )

    print("\n--- Betting simulation - All Top 5 Leagues ---")
    bets, wins, profit, roi, avg_odds = simulate_value_betting(
        t_probs_blend,
        raw_odds_arr,
        y_test_arr,
        edge_threshold=threshold,
        match_info=all_t_info if config.detailed_betting_log else None,
    )
    print(f"Ensemble Strategy (Edge > {threshold*100}%):")
    print(f"Total Bets Placed: {bets} (out of {len(X_test_arr)} matches)")
    if bets > 0:
        print(f"Won Bets: {wins} ({round(wins / bets * 100, 1)}% Hit Rate)")
        print(f"Average Odds Played: {round(avg_odds, 2)}")
    print(f"Net Profit: {round(profit, 2)} units")
    print(f"ROI: {round(roi, 2)}%")

        
    if config.print_verbose_audits:
        print_alignment_audit(
            probs=t_probs_blend,
            raw_odds=raw_odds_arr,
            y_true=y_test_arr,
            match_info=all_t_info,
            title="ALIGNMENT AUDIT (first 20 test matches)",
            n=20,
        )

        print_strategy_comparison(
            strategy_probs={
                "base": t_probs_model_arr,
                "market": t_mkt_fixed_arr,
                "meta": t_probs_meta,
                "mlp": t_probs_mlp,
                "logreg": t_probs_logreg,
                "met_no_market": t_probs_meta_nm,
                "mlp_no_market": t_probs_mlp_nm,
                "ensemble": t_probs_blend,
            },
            raw_odds=raw_odds_arr,
            y_true=y_test_arr,
            edge_threshold=threshold,
        )

        print_market_dependency_audit(
            y_true=y_test_arr,
            p_base=t_probs_model_arr,
            p_market=t_mkt_fixed_arr,
            p_meta=t_probs_meta,
            p_mlp=t_probs_mlp,
            p_ens=t_probs_blend,
        )

        print_profit_profile_audit(
            probs=t_probs_blend,
            raw_odds=raw_odds_arr,
            y_true=y_test_arr,
            match_info=all_t_info,
            edge_threshold=threshold,
        )

    for league in config.leagues:
        if league in league_best_params:
            league_best_params[league]["_blend_cfg"] = blend_cfg

    save_manifest(config.manifest_file, {
        "pipeline_version": PIPELINE_VERSION,
        "config": config.as_manifest(),
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "xgb": best_meta_cfg,
        "xgb_feature_columns": xgb_feature_columns,
        "xgb_n_features": len(xgb_feature_columns),
        "logreg": logreg_cfg,
        "logreg_feature_columns": logreg_feature_columns,
        "logreg_n_features": len(logreg_feature_columns),
        "mlp": best_mlp_cfg,
        "mlp_feature_columns": mlp_feature_columns,
        "mlp_n_features": len(mlp_feature_columns),
        "blend": blend_cfg,
        "selection_summary": {
            "winner_by_logloss": best_logloss["name"],
            "winner_by_logloss_value": round(best_logloss["logloss"], 6),
            "winner_by_betting_roi": best_roi["name"],
            "winner_by_betting_roi_value": round(best_roi["roi"], 4),
            "recommended_probability_model": best_logloss["name"],
            "recommended_betting_model": recommended_betting["name"],
            "recommended_betting_reason": betting_reason,
        },
    })

    generate_upcoming_matchday_picks(
        config.leagues,
        league_best_params,
        meta_final,
        best_meta_cfg,
        mlp_model,
        best_mlp_cfg,
        max_window_days=config.max_upcoming_window_days,
        pick_model=best_logloss["name"],
    )
    return {
        "league_best_params": league_best_params,
        "xgb_cfg": best_meta_cfg,
        "mlp_cfg": best_mlp_cfg,
        "blend_cfg": blend_cfg,
        "test_logloss": float(log_loss(y_test_arr, t_probs_blend)),
    }

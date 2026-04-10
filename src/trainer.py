from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from src.artifact_store import load_json_if_exists, load_pickle_if_exists, save_json, save_manifest, save_pickle
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
from src.feature_builder import build_meta_features, ensure_market_probs, time_split_val, market_probs_from_odds_row
from src.models.base import tune_league_params
from src.models.meta import (
    blend_probabilities,
    fit_xgb_model,
    make_mlp_pipeline,
    probs_from_meta_features,
    tune_blend_weights,
    tune_mlp_hyperparams,
    tune_xgb_hyperparams,
)
from src.reporting import print_per_league_test_metrics
from src.reporting_ext import print_confusion, print_prob_report
from src.services.upcoming import generate_upcoming_matchday_picks
from src.state_builder import streaming_block_probs_home_away

def run_training_pipeline(config: ExperimentConfig = DEFAULT_CONFIG):
    cached_params = load_json_if_exists(config.params_file) if config.use_cached_artifacts and not config.force_retune_leagues else None
    cached_meta = load_json_if_exists(config.meta_file) if config.use_cached_artifacts and not config.force_retune_meta else None
    cached_mlp = load_json_if_exists(config.mlp_meta_file) if config.use_cached_artifacts and not config.force_retune_mlp else None
    cached_blend = load_json_if_exists(config.blend_file) if config.use_cached_artifacts and not config.force_retune_blend else None
    league_best_params = {} if cached_params is None else cached_params
    all_X_early, all_y_early = [], []
    all_X_late, all_y_late = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []
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

        if league in league_best_params and not config.force_retune_leagues:
            params = league_best_params[league]
            print("\n--- Using cached params ---")
            print(f"K={params['K']}, ha={params['ha']}, beta={params['beta']}, decay={params['decay']}, rho={params['rho']}, T={params['T']}")
        else:
            params = tune_league_params(train_fit, val, df)
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

        v_probs_raw, v_y, v_mkt, v_aux, _ = streaming_block_probs_home_away(val, df, params["beta"], params["rho"], params["decay"], params["K"], params["ha"])
        v_probs_model = temperature_scale_probs(v_probs_raw, params["T"])
        all_X_val.extend(build_meta_features(v_probs_model, v_mkt, v_aux))
        all_y_val.extend(v_y)

        t_probs_raw, t_y, t_mkt, t_aux, t_raw_odds = streaming_block_probs_home_away(test, df, params["beta"], params["rho"], params["decay"], params["K"], params["ha"])
        t_probs_model = temperature_scale_probs(t_probs_raw, params["T"])
        t_mkt_fixed = ensure_market_probs(t_probs_model, t_mkt)

        all_X_test.extend(build_meta_features(t_probs_model, t_mkt_fixed, t_aux))
        all_y_test.extend(t_y)
        all_t_probs_model.extend(t_probs_model)
        all_t_mkt_fixed.extend(t_mkt_fixed)
        all_t_raw_odds.extend(t_raw_odds)
        all_t_info.extend(test[["date", "home_team", "away_team"]].to_dict("records"))
        per_league_test[league] = {"y": np.array(t_y, dtype=int), "p_model": np.array(t_probs_model, dtype=float), "p_mkt": np.array(t_mkt_fixed, dtype=float)}

    if cached_params is None or config.force_retune_leagues:
        save_json(config.params_file, league_best_params)
        print(f"\nSaved tuned league params to: {config.params_file}")

    X_early_arr, y_early_arr = np.array(all_X_early), np.array(all_y_early)
    X_late_arr, y_late_arr = np.array(all_X_late), np.array(all_y_late)
    X_val_arr, y_val_arr = np.array(all_X_val), np.array(all_y_val)
    X_test_arr, y_test_arr = np.array(all_X_test), np.array(all_y_test)
    t_probs_model_arr, t_mkt_fixed_arr = np.array(all_t_probs_model), np.array(all_t_mkt_fixed)
    raw_odds_arr = np.array(all_t_raw_odds)

    threshold = 0.05

        # Quick no-market ablation: drop market logits columns [3,4,5]
    NO_MARKET_COLS = [0, 1, 2, 6, 7, 8, 9, 10, 11]

    X_early_nm = X_early_arr[:, NO_MARKET_COLS]
    X_late_nm = X_late_arr[:, NO_MARKET_COLS]
    X_val_nm = X_val_arr[:, NO_MARKET_COLS]
    X_test_nm = X_test_arr[:, NO_MARKET_COLS]   

    recomputed_market = np.array(
        [market_probs_from_odds_row(o[0], o[1], o[2]) for o in raw_odds_arr],
        dtype=float,
    )

    mask = np.isfinite(recomputed_market).all(axis=1)
    max_diff = np.max(np.abs(recomputed_market[mask] - t_mkt_fixed_arr[mask]))
    print("SANITY max market diff:", max_diff)

    ev_check = recomputed_market[mask] * raw_odds_arr[mask] - 1.0
    print("SANITY max raw market EV:", np.max(ev_check))

    print("\n" + "=" * 50)
    print("=== META-MODEL Evaluation (XGBoost) ===")
    print("=" * 50)

    if cached_meta is not None and not config.force_retune_meta:
        best_meta_cfg = cached_meta
        print("Using cached XGBoost hyperparameters...")
    else:
        best_meta_cfg = tune_xgb_hyperparams(X_early_arr, y_early_arr, X_late_arr, y_late_arr)
        save_json(config.meta_file, best_meta_cfg)
        print(f"Saved XGBoost hyperparams to: {config.meta_file}")
    print(f"Best XGBoost Config -> LR: {best_meta_cfg['learning_rate']}, Depth: {best_meta_cfg['max_depth']}, Trees: {best_meta_cfg['n_estimators']}")
    print(f"Late VAL LogLoss: {round(best_meta_cfg['late_val_logloss'], 4)}")

    can_load_meta_model = config.use_cached_artifacts and config.model_file.exists() and not config.force_retune_meta and not config.force_refit_meta_model
    if can_load_meta_model:
        print(f"Loading trained XGBoost meta-model from: {config.model_file}")
        meta_final = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=config.random_state, n_jobs=-1)
        meta_final.load_model(str(config.model_file))
    else:
        print("Fitting final XGBoost meta-model...")
        meta_final = fit_xgb_model(X_val_arr, y_val_arr, best_meta_cfg)
        config.model_file.parent.mkdir(parents=True, exist_ok=True)
        meta_final.save_model(str(config.model_file))
        print(f"Saved trained XGBoost meta-model to: {config.model_file}")

    if cached_mlp is not None and not config.force_retune_mlp:
        best_mlp_cfg = cached_mlp
        print("Using cached MLP hyperparameters...")
    else:
        best_mlp_cfg = tune_mlp_hyperparams(X_early_arr, y_early_arr, X_late_arr, y_late_arr)
        save_json(config.mlp_meta_file, best_mlp_cfg)
        print(f"Saved MLP hyperparams to: {config.mlp_meta_file}")

    can_load_mlp_model = config.use_cached_artifacts and config.mlp_model_file.exists() and not config.force_retune_mlp and not config.force_refit_mlp_model
    if can_load_mlp_model:
        print(f"Loading trained MLP model from: {config.mlp_model_file}")
        mlp_model = load_pickle_if_exists(config.mlp_model_file)
    else:
        print("Fitting final MLP model...")
        mlp_model = make_mlp_pipeline(best_mlp_cfg)
        mlp_model.fit(X_val_arr, y_val_arr)
        save_pickle(config.mlp_model_file, mlp_model)
        print(f"Saved trained MLP model to: {config.mlp_model_file}")

    xgb_late_model = fit_xgb_model(X_early_arr, y_early_arr, best_meta_cfg)
    late_probs_xgb = xgb_late_model.predict_proba(X_late_arr)
    mlp_late_model = make_mlp_pipeline(best_mlp_cfg)
    mlp_late_model.fit(X_early_arr, y_early_arr)
    late_probs_mlp_raw = mlp_late_model.predict_proba(X_late_arr)
    late_probs_mlp = temperature_scale_probs(late_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    late_probs_base = probs_from_meta_features(X_late_arr, 0)
    late_probs_mkt = probs_from_meta_features(X_late_arr, 3)

    if cached_blend is not None and not config.force_retune_blend:
        blend_cfg = cached_blend
        print("Using cached blend weights...")
    else:
        blend_cfg = tune_blend_weights(y_late_arr, late_probs_base, late_probs_mkt, late_probs_xgb, late_probs_mlp, step=0.05)
        save_json(config.blend_file, blend_cfg)
        print(f"Saved blend weights to: {config.blend_file}")
    print(f"Blend weights: {blend_cfg['weights']}")
    print(f"Late VAL LogLoss: {round(blend_cfg['late_val_logloss'], 4)}")

    t_probs_meta = meta_final.predict_proba(X_test_arr)
    t_probs_mlp_raw = mlp_model.predict_proba(X_test_arr)
    t_probs_mlp = temperature_scale_probs(t_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    t_probs_blend = blend_probabilities(blend_cfg["weights"], {"base": t_probs_model_arr, "market": t_mkt_fixed_arr, "xgb": t_probs_meta, "mlp": t_probs_mlp})

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
    print_prob_report("DEEP LEARNING (MLP, calibrated)", t_probs_mlp, y_test_arr)
    print_prob_report("ENSEMBLE (XGB + MLP + Market + Base)", t_probs_blend, y_test_arr)
    print_confusion("ENSEMBLE", t_probs_blend, y_test_arr)
    print_per_league_test_metrics(list(config.leagues), per_league_test, t_probs_meta, t_probs_mlp, t_probs_blend)


    print("\n--- Betting simulation - All Top 5 Leagues ---")
    bets, wins, profit, roi, avg_odds = simulate_value_betting(t_probs_blend, raw_odds_arr, y_test_arr, edge_threshold=threshold, match_info=all_t_info)
    print(f"Ensemble Strategy (Edge > {threshold*100}%):")
    print(f"Total Bets Placed: {bets} (out of {len(X_test_arr)} matches)")
    if bets > 0:
        print(f"Won Bets: {wins} ({round(wins / bets * 100, 1)}% Hit Rate)")
        print(f"Average Odds Played: {round(avg_odds, 2)}")
    print(f"Net Profit: {round(profit, 2)} units")
    print(f"ROI: {round(roi, 2)}%")

        
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
        "config": config.as_manifest(),
        "feature_columns": [
            "model_logit_home", "model_logit_draw", "model_logit_away", "market_logit_home", "market_logit_draw", "market_logit_away", "elo_diff", "total_xg", "xg_diff", "mom_home", "mom_away", "mom_diff"
        ],
        "n_features": 12,
        "xgb": best_meta_cfg,
        "mlp": best_mlp_cfg,
        "blend": blend_cfg,
    })

    generate_upcoming_matchday_picks(config.leagues, league_best_params, meta_final, mlp_model, best_mlp_cfg, max_window_days=config.max_upcoming_window_days)
    return {
        "league_best_params": league_best_params,
        "xgb_cfg": best_meta_cfg,
        "mlp_cfg": best_mlp_cfg,
        "blend_cfg": blend_cfg,
        "test_logloss": float(log_loss(y_test_arr, t_probs_blend)),
    }

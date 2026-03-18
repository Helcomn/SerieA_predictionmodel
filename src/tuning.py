import numpy as np
import optuna
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.poisson_model import fit_team_strengths, fit_team_strengths_weighted, match_outcome_probs
from src.elo import compute_elo_ratings
from src.calibration import fit_temperature, temperature_scale_probs

# Απενεργοποίηση των περιττών logs του Optuna για καθαρό τερματικό
optuna.logging.set_verbosity(optuna.logging.WARNING)

def make_mlp_pipeline(best_mlp_cfg):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(best_mlp_cfg["hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            alpha=float(best_mlp_cfg["alpha"]),
            learning_rate_init=float(best_mlp_cfg["learning_rate_init"]),
            batch_size=64,
            max_iter=1500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
    )

def probs_from_meta_features(X_meta, start_idx):
    logits = X_meta[:, start_idx:start_idx + 3]
    probs = 1.0 / (1.0 + np.exp(-logits))
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums

def blend_probabilities(weight_dict, probs_dict):
    out = np.zeros_like(probs_dict["base"], dtype=float)
    total_w = 0.0
    for key, w in weight_dict.items():
        if key in probs_dict and probs_dict[key] is not None:
            out += float(w) * probs_dict[key]
            total_w += float(w)
    if total_w <= 0:
        out = probs_dict["xgb"].copy()
    else:
        out /= total_w
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums

def apply_blend(probs_xgb, probs_mlp, probs_market, probs_base, blend_cfg):
    return blend_probabilities(
        blend_cfg["weights"],
        {
            "base": probs_base,
            "market": probs_market,
            "xgb": probs_xgb,
            "mlp": probs_mlp,
        },
    )

def tune_league_params(train_fit, val, full_played_df, streaming_block_probs_home_away, apply_elo_to_lambdas):
    Ks = [40, 50, 60, 70]
    home_advs = [60, 80, 100, 110]
    betas = [0.10, 0.11, 0.12, 0.13]
    decays = [0.0005, 0.001, 0.002, 0.003]

    best = None
    print("\n--- Elo & Beta Tuning ---")
    from src.poisson_model import predict_lambdas
    import pandas as pd

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
        val_probs_raw, y_val_stream, _, _ = streaming_block_probs_home_away(
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

def tune_xgb_hyperparams(X_early, y_early, X_late, y_late, n_trials=20):
    print("Tuning XGBoost Hyperparameters using Optuna (Bayesian Optimization). Please wait...")
    
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
        md = trial.suggest_int("max_depth", 2, 6)
        ne = trial.suggest_int("n_estimators", 50, 600)
        
        meta = XGBClassifier(
            n_estimators=ne, learning_rate=lr, max_depth=md,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
        meta.fit(X_early, y_early)
        late_probs = meta.predict_proba(X_late)
        return log_loss(y_late, late_probs)
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    return {
        "learning_rate": float(study.best_params["learning_rate"]),
        "max_depth": int(study.best_params["max_depth"]),
        "n_estimators": int(study.best_params["n_estimators"]),
        "late_val_logloss": float(study.best_value)
    }

def tune_mlp_hyperparams(X_early, y_early, X_late, y_late, n_trials=15):
    print("Tuning MLP Hyperparameters using Optuna. Please wait...")
    
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 2)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128]))
        
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        lr_init = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
        
        cfg = {
            "hidden_layer_sizes": tuple(layers),
            "alpha": alpha,
            "learning_rate_init": lr_init
        }
        
        model = make_mlp_pipeline(cfg)
        model.fit(X_early, y_early)
        late_probs_raw = model.predict_proba(X_late)
        
        T_mlp = fit_temperature(late_probs_raw, y_late)
        late_probs_cal = temperature_scale_probs(late_probs_raw, T_mlp)
        
        return log_loss(y_late, late_probs_cal)
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_layers = [study.best_params[f"n_units_l{i}"] for i in range(study.best_params["n_layers"])]
    best_cfg = {
        "hidden_layer_sizes": tuple(best_layers),
        "alpha": study.best_params["alpha"],
        "learning_rate_init": study.best_params["learning_rate_init"]
    }
    
    # Επανεκπαίδευση με τις καλύτερες παραμέτρους για να βρούμε την τελική θερμοκρασία (Temperature)
    best_model = make_mlp_pipeline(best_cfg)
    best_model.fit(X_early, y_early)
    best_probs_raw = best_model.predict_proba(X_late)
    best_T = fit_temperature(best_probs_raw, y_late)
    
    return {
        "hidden_layer_sizes": best_layers,
        "alpha": float(best_cfg["alpha"]),
        "learning_rate_init": float(best_cfg["learning_rate_init"]),
        "late_val_logloss": float(study.best_value),
        "temperature": float(best_T)
    }

def tune_blend_weights(y_late, probs_base, probs_market, probs_xgb, probs_mlp, step=0.1):
    weight_grid = np.arange(0.0, 1.0 + 1e-9, step)
    best = None
    print("Tuning blend weights on late validation. Please wait...")
    for wb in weight_grid:
        for wm in weight_grid:
            for wx in weight_grid:
                for wn in weight_grid:
                    if wb + wm + wx + wn == 0:
                        continue
                    weights = {"base": float(wb), "market": float(wm), "xgb": float(wx), "mlp": float(wn)}
                    probs_blend = blend_probabilities(weights, {"base": probs_base, "market": probs_market, "xgb": probs_xgb, "mlp": probs_mlp})
                    ll = log_loss(y_late, probs_blend)
                    if best is None or ll < best["late_val_logloss"]:
                        best = {"weights": weights, "late_val_logloss": float(ll)}
    return best
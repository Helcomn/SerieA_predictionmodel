from __future__ import annotations

import numpy as np
try:
    import optuna
except ImportError:  # fallback for environments without optuna
    optuna = None
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.calibration import fit_temperature, temperature_scale_probs

if optuna is not None:
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
        ),
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


def tune_xgb_hyperparams(X_early, y_early, X_late, y_late, n_trials=20):
    print("Tuning XGBoost Hyperparameters. Please wait...")

    def score_cfg(lr, md, ne):
        meta = XGBClassifier(
            n_estimators=ne, learning_rate=lr, max_depth=md,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
        meta.fit(X_early, y_early)
        return log_loss(y_late, meta.predict_proba(X_late))

    if optuna is not None:
        def objective(trial):
            lr = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
            md = trial.suggest_int("max_depth", 2, 6)
            ne = trial.suggest_int("n_estimators", 50, 600)
            return score_cfg(lr, md, ne)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return {
            "learning_rate": float(study.best_params["learning_rate"]),
            "max_depth": int(study.best_params["max_depth"]),
            "n_estimators": int(study.best_params["n_estimators"]),
            "late_val_logloss": float(study.best_value),
        }

    grid = [(lr, md, ne) for lr in [0.01, 0.02, 0.05, 0.1] for md in [2, 3, 4, 5] for ne in [50, 100, 200, 400]]
    best = None
    for lr, md, ne in grid[:max(8, min(len(grid), n_trials * 3))]:
        ll = score_cfg(lr, md, ne)
        if best is None or ll < best[0]:
            best = (ll, lr, md, ne)
    return {"learning_rate": float(best[1]), "max_depth": int(best[2]), "n_estimators": int(best[3]), "late_val_logloss": float(best[0])}


def fit_xgb_model(X_train, y_train, cfg):
    model = XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=float(cfg["learning_rate"]),
        max_depth=int(cfg["max_depth"]),
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def tune_mlp_hyperparams(X_early, y_early, X_late, y_late, n_trials=15):
    print("Tuning MLP Hyperparameters. Please wait...")

    def score_cfg(cfg):
        model = make_mlp_pipeline(cfg)
        model.fit(X_early, y_early)
        late_probs_raw = model.predict_proba(X_late)
        T_mlp = fit_temperature(late_probs_raw, y_late)
        late_probs_cal = temperature_scale_probs(late_probs_raw, T_mlp)
        return log_loss(y_late, late_probs_cal), T_mlp

    if optuna is not None:
        def objective(trial):
            n_layers = trial.suggest_int("n_layers", 1, 2)
            layers = [trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128]) for i in range(n_layers)]
            alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
            lr_init = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
            cfg = {"hidden_layer_sizes": tuple(layers), "alpha": alpha, "learning_rate_init": lr_init}
            ll, _ = score_cfg(cfg)
            return ll

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        best_layers = [study.best_params[f"n_units_l{i}"] for i in range(study.best_params["n_layers"])]
        best_cfg = {
            "hidden_layer_sizes": best_layers,
            "alpha": float(study.best_params["alpha"]),
            "learning_rate_init": float(study.best_params["learning_rate_init"]),
        }
        best_ll, best_T = score_cfg(best_cfg)
        best_cfg["late_val_logloss"] = float(best_ll)
        best_cfg["temperature"] = float(best_T)
        return best_cfg

    grid = []
    for layers in ([32], [64], [128], [64, 32], [128, 64]):
        for alpha in [1e-5, 1e-4, 1e-3]:
            for lr in [1e-4, 5e-4, 1e-3, 2e-3]:
                grid.append({"hidden_layer_sizes": layers, "alpha": alpha, "learning_rate_init": lr})
    best = None
    for cfg in grid[:max(8, min(len(grid), n_trials * 3))]:
        ll, T = score_cfg(cfg)
        if best is None or ll < best[0]:
            best = (ll, T, cfg)
    out = dict(best[2])
    out["late_val_logloss"] = float(best[0])
    out["temperature"] = float(best[1])
    return out


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

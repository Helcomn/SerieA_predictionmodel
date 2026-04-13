from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "baseline_xgboost_v3_formpoints"
    artifacts_dir: Path = Path("artifacts")
    train_cut: str = "2024-07-01"
    test_cut: str = "2025-07-01"
    leagues: tuple[str, ...] = ("england", "spain", "italy", "germany", "france")
    use_cached_artifacts: bool = True
    force_retune_leagues: bool = False
    force_retune_meta: bool = False
    force_refit_meta_model: bool = False
    force_retune_mlp: bool = False
    force_refit_mlp_model: bool = False
    force_retune_blend: bool = False
    random_state: int = 42
    max_upcoming_window_days: int = 4
    detailed_betting_log: bool = False
    print_verbose_audits: bool = False

    @property
    def params_file(self) -> Path:
        return self.artifacts_dir / f"best_params_{self.experiment_name}.json"

    @property
    def meta_file(self) -> Path:
        return self.artifacts_dir / f"best_meta_{self.experiment_name}.json"

    @property
    def model_file(self) -> Path:
        return self.artifacts_dir / f"meta_model_{self.experiment_name}.json"

    @property
    def mlp_meta_file(self) -> Path:
        return self.artifacts_dir / f"best_mlp_{self.experiment_name}.json"

    @property
    def mlp_model_file(self) -> Path:
        return self.artifacts_dir / f"mlp_model_{self.experiment_name}.pkl"

    @property
    def blend_file(self) -> Path:
        return self.artifacts_dir / f"best_blend_{self.experiment_name}.json"

    @property
    def manifest_file(self) -> Path:
        return self.artifacts_dir / f"manifest_{self.experiment_name}.json"

    @property
    def results_csv_file(self) -> Path:
        return self.artifacts_dir / "experiment_results.csv"

    @property
    def ablations_csv_file(self) -> Path:
        return self.artifacts_dir / "feature_ablations.csv"

    def as_manifest(self) -> Dict:
        data = asdict(self)
        data["artifacts_dir"] = str(self.artifacts_dir)
        data["leagues"] = list(self.leagues)
        return data


DEFAULT_CONFIG = ExperimentConfig()

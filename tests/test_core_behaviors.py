import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

import src.data_processing as data_processing
from src.evaluation import betting_records, simulate_value_betting
from src.feature_builder import ensure_market_probs, market_probs_from_odds_row
from src.models.meta import blend_probabilities
from src.state_builder import EXTRA_AUX_LEN, compute_pre_match_extra_features
from src.team_names import normalize_team_name
from src.understat_data import add_understat_xg


class FeatureBuilderTests(unittest.TestCase):
    def test_market_probs_normalize_valid_odds(self):
        probs = market_probs_from_odds_row(2.0, 3.5, 4.0)

        self.assertTrue(np.isfinite(probs).all())
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_market_probs_returns_nan_for_invalid_odds(self):
        probs = market_probs_from_odds_row(2.0, np.nan, 4.0)

        self.assertTrue(np.isnan(probs).all())

    def test_missing_market_probs_fall_back_to_model_probs(self):
        model = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
        market = np.array([[np.nan, np.nan, np.nan], [0.3, 0.3, 0.4]])

        fixed = ensure_market_probs(model, market)

        np.testing.assert_allclose(fixed[0], model[0])
        np.testing.assert_allclose(fixed[1], market[1])


class TeamNameTests(unittest.TestCase):
    def test_fixture_team_aliases_match_historical_names(self):
        self.assertEqual(normalize_team_name("FC Barcelona", "spain"), "Barcelona")
        self.assertEqual(normalize_team_name("Rayo Vallecano", "spain"), "Vallecano")
        self.assertEqual(normalize_team_name("Borussia Dortmund", "germany"), "Dortmund")
        self.assertEqual(normalize_team_name("1. FC Heidenheim 1846", "germany"), "Heidenheim")

    def test_loader_normalizes_downloaded_fixture_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw" / "spain"
            raw_dir.mkdir(parents=True)
            pd.DataFrame([
                {
                    "Date": "01/01/2026",
                    "HomeTeam": "Barcelona",
                    "AwayTeam": "Alaves",
                    "FTHG": 2,
                    "FTAG": 0,
                },
                {
                    "Date": "08/01/2026",
                    "HomeTeam": "Alaves",
                    "AwayTeam": "FC Barcelona",
                    "FTHG": np.nan,
                    "FTAG": np.nan,
                },
            ]).to_csv(raw_dir / "SP1_2025_fixtures.csv", index=False)

            original_root = data_processing.PROJECT_ROOT
            original_add_understat_xg = data_processing.add_understat_xg
            try:
                data_processing.PROJECT_ROOT = root
                data_processing.add_understat_xg = lambda df, league_name: df
                loaded = data_processing.load_league_data("spain")
            finally:
                data_processing.PROJECT_ROOT = original_root
                data_processing.add_understat_xg = original_add_understat_xg

        self.assertIn("Barcelona", set(loaded["away_team"]))
        self.assertNotIn("FC Barcelona", set(loaded["away_team"]))


class BettingSimulationTests(unittest.TestCase):
    def test_betting_simulation_can_run_silently(self):
        probs = np.array([[0.55, 0.25, 0.20]])
        odds = np.array([[2.1, 3.2, 4.5]])
        y_true = np.array([0])

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = simulate_value_betting(probs, odds, y_true, verbose=False)

        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 1)

    def test_betting_records_include_open_to_close_value_when_available(self):
        probs = np.array([[0.55, 0.25, 0.20]])
        odds = np.array([[2.1, 3.2, 4.5]])
        y_true = np.array([0])
        match_info = [{
            "date": pd.Timestamp("2025-01-01"),
            "league": "england",
            "home_team": "A",
            "away_team": "B",
            "open_odds_home": 2.1,
            "close_odds_home": 2.0,
        }]

        records = betting_records(probs, odds, y_true, match_info=match_info)

        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(float(records.loc[0, "clv_decimal"]), 0.05)

    def test_betting_records_leave_clv_blank_without_opening_price(self):
        probs = np.array([[0.55, 0.25, 0.20]])
        odds = np.array([[2.1, 3.2, 4.5]])
        y_true = np.array([0])
        match_info = [{
            "date": pd.Timestamp("2025-01-01"),
            "league": "england",
            "home_team": "A",
            "away_team": "B",
            "close_odds_home": 2.0,
        }]

        records = betting_records(probs, odds, y_true, match_info=match_info)

        self.assertEqual(len(records), 1)
        self.assertTrue(np.isnan(float(records.loc[0, "clv_decimal"])))


class BlendTests(unittest.TestCase):
    def test_blend_probabilities_renormalizes_output(self):
        base = np.array([[0.6, 0.2, 0.2]])
        xgb = np.array([[0.3, 0.4, 0.3]])

        blended = blend_probabilities(
            {"base": 2.0, "xgb": 1.0},
            {"base": base, "xgb": xgb},
        )

        self.assertAlmostEqual(float(blended.sum()), 1.0)
        self.assertEqual(blended.shape, (1, 3))


class RollingFeatureTests(unittest.TestCase):
    def test_pre_match_features_use_only_past_matches(self):
        past = pd.DataFrame([
            {
                "date": pd.Timestamp("2024-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_shots": 10,
                "away_shots": 5,
                "home_shots_target": 4,
                "away_shots_target": 2,
                "home_corners": 6,
                "away_corners": 3,
                "home_yellows": 1,
                "away_yellows": 2,
                "home_reds": 0,
                "away_reds": 1,
            },
        ])
        row = pd.Series({
            "home_team": "A",
            "away_team": "B",
            "open_odds_home": 2.0,
            "open_odds_draw": 3.0,
            "open_odds_away": 4.0,
            "close_odds_home": 1.8,
            "close_odds_draw": 3.2,
            "close_odds_away": 4.5,
            "ou25_over_prob": 0.57,
            "ah_line": -0.5,
        })

        features = compute_pre_match_extra_features(row, past)

        self.assertEqual(len(features), EXTRA_AUX_LEN)
        self.assertEqual(features[0], 10.0)
        self.assertEqual(features[1], 5.0)
        self.assertEqual(features[2], 5.0)
        self.assertAlmostEqual(features[15], 0.57)
        self.assertAlmostEqual(features[16], -0.5)

    def test_loader_extracts_over_under_25_odds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw" / "england"
            raw_dir.mkdir(parents=True)
            pd.DataFrame([{
                "Date": "01/01/2025",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 2,
                "FTAG": 1,
                "B365H": 2.0,
                "B365D": 3.5,
                "B365A": 4.0,
                "AvgC>2.5": 1.8,
                "AvgC<2.5": 2.1,
            }]).to_csv(raw_dir / "E0_2024.csv", index=False)

            original_root = data_processing.PROJECT_ROOT
            original_add_understat_xg = data_processing.add_understat_xg
            try:
                data_processing.PROJECT_ROOT = root
                data_processing.add_understat_xg = lambda df, league_name: df
                loaded = data_processing.load_league_data("england")
            finally:
                data_processing.PROJECT_ROOT = original_root
                data_processing.add_understat_xg = original_add_understat_xg

        self.assertAlmostEqual(float(loaded.loc[0, "ou25_over_odds"]), 1.8)
        self.assertAlmostEqual(float(loaded.loc[0, "ou25_under_odds"]), 2.1)
        self.assertTrue(np.isfinite(float(loaded.loc[0, "ou25_over_prob"])))

    def test_understat_match_rows_join_to_loaded_matches(self):
        base = pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"),
            "home_team": "Arsenal",
            "away_team": "Chelsea",
        }])
        understat = pd.DataFrame([{
            "date": "2024-01-01",
            "league": "EPL",
            "team_h": "Arsenal",
            "team_a": "Chelsea",
            "h_xg": 1.7,
            "a_xg": 0.8,
            "h_npxg": 1.4,
            "a_npxg": 0.7,
            "h_xpts": 2.1,
            "a_xpts": 0.6,
        }])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "understat_matches.csv"
            understat.to_csv(path, index=False)

            merged = add_understat_xg(base, "england", path=path)

        self.assertAlmostEqual(float(merged.loc[0, "home_understat_xg"]), 1.7)
        self.assertAlmostEqual(float(merged.loc[0, "away_understat_npxg"]), 0.7)


if __name__ == "__main__":
    unittest.main()

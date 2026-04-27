from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNDERSTAT_MATCHES_FILE = PROJECT_ROOT / "data" / "external" / "understat_matches.csv"

LEAGUE_ALIASES = {
    "england": {"epl", "premier league", "english premier league"},
    "spain": {"la liga", "laliga", "spain"},
    "italy": {"serie a", "italy"},
    "germany": {"bundesliga", "germany"},
    "france": {"ligue 1", "france"},
}

TEAM_ALIASES = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "man city": "manchester city",
    "newcastle": "newcastle united",
    "tottenham": "tottenham hotspur",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "west brom": "west bromwich albion",
    "brighton": "brighton and hove albion",
    "sheffield united": "sheffield utd",
    "sheffield utd": "sheffield utd",
    "nottm forest": "nottingham forest",
    "nottingham forest": "nottm forest",
    "leeds": "leeds united",
    "leicester": "leicester city",
    "norwich": "norwich city",
    "cardiff": "cardiff city",
    "stoke": "stoke city",
    "swansea": "swansea city",
    "hull": "hull city",
    "qpr": "queens park rangers",
    "west ham": "west ham united",
    "bournemouth": "afc bournemouth",
    "hertha berlin": "hertha bsc",
    "bayern munich": "bayern",
    "bayern munchen": "bayern",
    "monchengladbach": "borussia m gladbach",
    "m gladbach": "borussia m gladbach",
    "gladbach": "borussia m gladbach",
    "koln": "cologne",
    "fc koln": "cologne",
    "schalke 04": "schalke",
    "mainz 05": "mainz",
    "rb leipzig": "rasenballsport leipzig",
    "leverkusen": "bayer leverkusen",
    "ein frankfurt": "eintracht frankfurt",
    "hannover": "hannover 96",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "ac milan": "milan",
    "as roma": "roma",
    "roma": "roma",
    "hellas verona": "verona",
    "spal": "spal 2013",
    "huesca": "sd huesca",
    "valladolid": "real valladolid",
    "real valladolid": "real valladolid",
    "espanol": "espanyol",
    "sp gijon": "sporting gijon",
    "athletic bilbao": "athletic club",
    "ath bilbao": "athletic club",
    "atletico madrid": "atletico",
    "atl madrid": "atletico",
    "ath madrid": "atletico",
    "real betis": "betis",
    "betis": "betis",
    "celta vigo": "celta",
    "deportivo la coruna": "deportivo",
    "la coruna": "deportivo",
    "espanyol": "espanyol",
    "rayo vallecano": "rayo",
    "vallecano": "rayo",
    "real sociedad": "sociedad",
    "sociedad": "sociedad",
    "sporting gijon": "sporting gijon",
    "paris saint germain": "psg",
    "paris sg": "psg",
    "st etienne": "saint etienne",
    "saint etienne": "saint etienne",
    "clermont": "clermont foot",
    "nimes": "nimes olympique",
    "ajaccio gfco": "gfc ajaccio",
    "amiens": "amiens sc",
    "hertha": "hertha bsc",
    "fortuna dusseldorf": "fortuna duesseldorf",
    "greuther furth": "greuther fuerth",
    "hamburg": "hamburger",
    "nurnberg": "nuernberg",
    "bielefeld": "arminia bielefeld",
}

UNDERSTAT_VALUE_COLUMNS = [
    "home_understat_xg",
    "away_understat_xg",
    "home_understat_npxg",
    "away_understat_npxg",
    "home_understat_xpts",
    "away_understat_xpts",
]


def normalize_team_name(name) -> str:
    if not isinstance(name, str):
        return ""
    text = name.lower()
    text = text.replace("&", "and")
    text = re.sub(r"\b(fc|cf|afc|ssc|ac|as|sc|sv|vfb|borussia)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    normalized = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(normalized, normalized)


def _normalise_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _filter_league(df: pd.DataFrame, league_name: str) -> pd.DataFrame:
    if "league" not in df.columns:
        return df
    aliases = LEAGUE_ALIASES.get(league_name, {league_name})
    league_norm = df["league"].astype(str).str.lower().str.strip()
    return df[league_norm.isin(aliases)].copy()


def _first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _from_match_rows(raw: pd.DataFrame, league_name: str) -> pd.DataFrame | None:
    columns = set(raw.columns)
    date_col = _first_existing(columns, ["date", "Date"])
    home_col = _first_existing(columns, ["team_h", "home_team", "home", "h_team"])
    away_col = _first_existing(columns, ["team_a", "away_team", "away", "a_team"])
    hxg_col = _first_existing(columns, ["h_xg", "home_xg", "xG_home"])
    axg_col = _first_existing(columns, ["a_xg", "away_xg", "xG_away"])

    if not all([date_col, home_col, away_col, hxg_col, axg_col]):
        return None

    df = _filter_league(raw, league_name)
    out = pd.DataFrame({
        "date": _normalise_date(df[date_col]),
        "home_key": df[home_col].map(normalize_team_name),
        "away_key": df[away_col].map(normalize_team_name),
        "home_understat_xg": pd.to_numeric(df[hxg_col], errors="coerce"),
        "away_understat_xg": pd.to_numeric(df[axg_col], errors="coerce"),
    })

    hnpxg = _first_existing(columns, ["h_npxg", "home_npxg", "npxG_home"])
    anpxg = _first_existing(columns, ["a_npxg", "away_npxg", "npxG_away"])
    hxpts = _first_existing(columns, ["h_xpts", "home_xpts", "xpts_home"])
    axpts = _first_existing(columns, ["a_xpts", "away_xpts", "xpts_away"])
    out["home_understat_npxg"] = pd.to_numeric(df[hnpxg], errors="coerce") if hnpxg else np.nan
    out["away_understat_npxg"] = pd.to_numeric(df[anpxg], errors="coerce") if anpxg else np.nan
    out["home_understat_xpts"] = pd.to_numeric(df[hxpts], errors="coerce") if hxpts else np.nan
    out["away_understat_xpts"] = pd.to_numeric(df[axpts], errors="coerce") if axpts else np.nan
    return out.dropna(subset=["date", "home_key", "away_key"])


def _from_team_rows(raw: pd.DataFrame, league_name: str) -> pd.DataFrame | None:
    columns = set(raw.columns)
    date_col = _first_existing(columns, ["date", "Date"])
    team_col = _first_existing(columns, ["club_name", "team", "Team", "team_name"])
    side_col = _first_existing(columns, ["home_away", "side", "h_a"])
    xg_col = _first_existing(columns, ["xG", "xg"])

    if not all([date_col, team_col, side_col, xg_col]):
        return None

    df = _filter_league(raw, league_name).copy()
    df["date"] = _normalise_date(df[date_col])
    df["team_key"] = df[team_col].map(normalize_team_name)
    side = df[side_col].astype(str).str.lower().str.strip()
    df["is_home"] = side.isin(["h", "home"])
    df["is_away"] = side.isin(["a", "away"])

    npxg_col = _first_existing(columns, ["npxG", "npxg"])
    xpts_col = _first_existing(columns, ["xpts", "xPTS"])
    xga_col = _first_existing(columns, ["xGA", "xga"])
    scored_col = _first_existing(columns, ["scored", "goals_for"])
    missed_col = _first_existing(columns, ["missed", "goals_against"])
    df["understat_xg"] = pd.to_numeric(df[xg_col], errors="coerce")
    df["understat_xga"] = pd.to_numeric(df[xga_col], errors="coerce") if xga_col else np.nan
    df["understat_npxg"] = pd.to_numeric(df[npxg_col], errors="coerce") if npxg_col else np.nan
    df["understat_xpts"] = pd.to_numeric(df[xpts_col], errors="coerce") if xpts_col else np.nan
    df["scored"] = pd.to_numeric(df[scored_col], errors="coerce") if scored_col else np.nan
    df["missed"] = pd.to_numeric(df[missed_col], errors="coerce") if missed_col else np.nan

    home = df[df["is_home"]][["date", "team_key", "understat_xg", "understat_xga", "understat_npxg", "understat_xpts", "scored", "missed"]].rename(columns={
        "team_key": "home_key",
        "understat_xg": "home_understat_xg",
        "understat_xga": "home_understat_xga",
        "understat_npxg": "home_understat_npxg",
        "understat_xpts": "home_understat_xpts",
        "scored": "home_scored",
        "missed": "home_missed",
    })
    away = df[df["is_away"]][["date", "team_key", "understat_xg", "understat_xga", "understat_npxg", "understat_xpts", "scored", "missed"]].rename(columns={
        "team_key": "away_key",
        "understat_xg": "away_understat_xg",
        "understat_xga": "away_understat_xga",
        "understat_npxg": "away_understat_npxg",
        "understat_xpts": "away_understat_xpts",
        "scored": "away_scored",
        "missed": "away_missed",
    })
    out = home.merge(away, on="date", how="inner")
    if {"home_scored", "home_missed", "away_scored", "away_missed"}.issubset(out.columns):
        out = out[(out["home_scored"] == out["away_missed"]) & (out["home_missed"] == out["away_scored"])]
    if {"home_understat_xg", "home_understat_xga", "away_understat_xg", "away_understat_xga"}.issubset(out.columns):
        out = out[
            np.isclose(out["home_understat_xg"], out["away_understat_xga"], atol=1e-4, equal_nan=False)
            & np.isclose(out["away_understat_xg"], out["home_understat_xga"], atol=1e-4, equal_nan=False)
        ]
    out = out.drop(columns=[
        "home_understat_xga",
        "away_understat_xga",
        "home_scored",
        "home_missed",
        "away_scored",
        "away_missed",
    ], errors="ignore")
    return out.dropna(subset=["date", "home_key", "away_key"])


def add_understat_xg(df: pd.DataFrame, league_name: str, path: Path = UNDERSTAT_MATCHES_FILE) -> pd.DataFrame:
    out = df.copy()
    for col in UNDERSTAT_VALUE_COLUMNS:
        out[col] = np.nan

    if not path.exists():
        return out

    raw = pd.read_csv(path)
    understat = _from_match_rows(raw, league_name)
    if understat is None:
        understat = _from_team_rows(raw, league_name)
    if understat is None or understat.empty:
        return out

    out["date_key"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["home_key"] = out["home_team"].map(normalize_team_name)
    out["away_key"] = out["away_team"].map(normalize_team_name)

    understat = understat.drop_duplicates(["date", "home_key", "away_key"], keep="first")
    merged = out.merge(
        understat,
        left_on=["date_key", "home_key", "away_key"],
        right_on=["date", "home_key", "away_key"],
        how="left",
        suffixes=("", "_understat_join"),
    )

    for col in UNDERSTAT_VALUE_COLUMNS:
        join_col = f"{col}_understat_join"
        if join_col in merged.columns:
            merged[col] = merged[join_col]

    drop_cols = [c for c in merged.columns if c.endswith("_understat_join")]
    drop_cols.extend(["date_key", "home_key", "away_key"])
    return merged.drop(columns=drop_cols, errors="ignore")


def load_understat_matches_for_league(league_name: str, path: Path = UNDERSTAT_MATCHES_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    understat = _from_match_rows(raw, league_name)
    if understat is None:
        understat = _from_team_rows(raw, league_name)
    if understat is None:
        return pd.DataFrame()
    return understat.drop_duplicates(["date", "home_key", "away_key"], keep="first")


def understat_coverage_report(fixtures_df: pd.DataFrame, league_name: str, path: Path = UNDERSTAT_MATCHES_FILE) -> dict:
    played = fixtures_df[fixtures_df["is_played"] == True].copy()
    if played.empty:
        return {"league": league_name, "played": 0, "matched": 0, "coverage": 0.0, "unmatched_teams": []}

    understat = load_understat_matches_for_league(league_name, path)
    if understat.empty:
        return {"league": league_name, "played": len(played), "matched": 0, "coverage": 0.0, "unmatched_teams": sorted(set(played["home_team"]) | set(played["away_team"]))}

    min_date = understat["date"].min()
    max_date = understat["date"].max()
    played_overlap = played[
        (pd.to_datetime(played["date"], errors="coerce").dt.normalize() >= min_date)
        & (pd.to_datetime(played["date"], errors="coerce").dt.normalize() <= max_date)
    ].copy()
    if played_overlap.empty:
        return {"league": league_name, "played": len(played), "matched": 0, "coverage": 0.0, "unmatched_teams": [], "understat_min_date": min_date, "understat_max_date": max_date}

    probe = played_overlap[["date", "home_team", "away_team"]].copy()
    probe["date_key"] = pd.to_datetime(probe["date"], errors="coerce").dt.normalize()
    probe["home_key"] = probe["home_team"].map(normalize_team_name)
    probe["away_key"] = probe["away_team"].map(normalize_team_name)
    joined = probe.merge(
        understat[["date", "home_key", "away_key"]],
        left_on=["date_key", "home_key", "away_key"],
        right_on=["date", "home_key", "away_key"],
        how="left",
        indicator=True,
    )
    matched_mask = joined["_merge"] == "both"
    unmatched = joined[~matched_mask]
    unmatched_teams = sorted(set(unmatched["home_team"]) | set(unmatched["away_team"]))
    return {
        "league": league_name,
        "played": int(len(played_overlap)),
        "matched": int(matched_mask.sum()),
        "coverage": float(matched_mask.mean()),
        "unmatched_teams": unmatched_teams,
        "understat_min_date": min_date,
        "understat_max_date": max_date,
    }

"""
Helpers for metal-peer-risk-scoring notebooks.

Data loading, peer-group scoring, simulation, and ML model comparison.
"""

from __future__ import annotations

import csv
import re
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_CSV_COLUMNS: tuple[str, ...] = (
    "policy_sk",
    "policy_no",
    "effective_dt",
    "expiration_dt",
    "product_cd",
    "risk_state_cd",
    "insured_nm",
    "insured_type",
    "policy_status",
    "program_type",
    "current_underwriter_nm",
    "current_producer_nm",
    "lifetime_claim_ct",
    "lifetime_loss_incurred_amt",
    "home_coverage_sk",
    "policy_history_sk",
    "transaction_seq_no",
    "residence_type",
    "dwelling_limit_amt",
    "other_structures_limit_amt",
    "contents_limit_amt",
    "loss_of_use_limit_amt",
    "personal_liability_limit_amt",
    "medical_payments_limit_amt",
    "total_insured_value_amt",
    "aop_deductible",
    "water_deductible",
    "hurricane_deductible",
    "hurricane_or_named_storm_deductible",
    "named_storm_deductible",
    "tornado_or_hailstorm_deductible",
    "wind_or_hailstorm_deductible",
    "wind_derived_deductible",
    "wildfire_deductible",
    "prior_claim_last5yr_in",
    "prior_nonwater_claim_ct",
    "prior_water_claim_ct",
    "rating_territory_cd",
    "distance_to_coast",
    "distance_to_shore",
    "within_500feet_from_shore_in",
    "distance_to_fire_hydrant_feet",
    "distance_to_fire_station_miles",
    "fire_protection",
    "protection_class",
    "earthquake_zone",
    "windborne_debris_region_in",
    "windpool_eligibility_in",
    "sinkhole_risk_level",
    "occupancy_type",
    "total_finished_square_feet",
    "construction_type",
    "basement_type",
    "active_renovation",
    "built_year",
    "electrical_updated_year",
    "hvac_updated_year",
    "plumbing_updated_year",
    "roof_updated_year",
    "no_of_stories",
    "roof_covering",
    "roof_geometry",
    "roof_deck_attachment",
    "roof_wall_attachment",
    "foundation_type",
    "no_of_bathrooms",
    "no_of_fireplaces",
    "rate_on_line",
    "rate_on_line_exclude_collection",
    "aon_hurricane_cat_score_amt",
    "aon_hurricane_reinsurance_premium_amt",
    "aon_hurricane_cat_score_to_premium_ratio",
    "aon_hurricane_aal_to_premium_ratio",
    "aon_hurricane_aal_amt",
    "aon_hurricane_reinsurance_margin_amt",
    "aon_hurricane_capital_cost_amt",
    "wildfire_score",
    "wildfire_risk_score",
    "wildfire_risk_class",
    "facultative_ceded_premium_amt",
    "home_additional_coverage_sk",
    "water_leak_detection_system",
    "central_reporting_fire_alarm_in",
    "central_reporting_burglar_alarm_in",
    "residential_sprinkler_system_in",
    "backup_generator_in",
    "guard_community_patrol_service_in",
    "gated_community_patrol_service",
    "wildfire_protection_enrollment_in",
    "roof_exclusion_in",
    "waterdamage_exclusion_in",
    "wind_hail_exclusion_in",
    "risk_score_water_non_weather",
    "risk_score_water_weather",
    "risk_score_water_backup",
    "risk_score_wind_hail",
    "risk_score_other",
    "risk_score_lightning",
    "risk_score_theft",
    "risk_score_liability",
    "risk_score_hurricane",
    "risk_score_wildfire",
    "risk_score_sinkhole_mine",
    "risk_score_all_perils",
    "risk_score_fire",
    "earthquake_score",
    "property_age_years",
    "roof_age_years",
    "exposure_band",
    "has_claim_history",
    "demo_risk_band",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
DEFAULT_CSV = PROJECT_ROOT / "Results.csv"

# ---------------------------------------------------------------------------
# Config (notebook scoring parameters)
# ---------------------------------------------------------------------------

PEER_GROUP_COLUMNS = ("risk_state_cd", "program_type", "product_cd", "occupancy_type")

COMPONENT_WEIGHTS = {
    "exposure": 0.25,
    "geography_hazard": 0.25,
    "building": 0.20,
    "claims": 0.20,
    "protection": 0.10,
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "exposure": [
        "total_insured_value_amt",
        "dwelling_limit_amt",
        "contents_limit_amt",
        "total_finished_square_feet",
    ],
    "geography_hazard": [
        "distance_to_coast",
        "distance_to_shore",
        "wildfire_risk_score",
        "wildfire_score",
    ],
    "building": [
        "built_year",
        "roof_updated_year",
        "property_age_years",
        "roof_age_years",
        "no_of_stories",
    ],
    "claims": [
        "lifetime_claim_ct",
        "lifetime_loss_incurred_amt",
        "prior_water_claim_ct",
        "prior_nonwater_claim_ct",
    ],
    "protection": [
        "distance_to_fire_hydrant_feet",
        "distance_to_fire_station_miles",
    ],
}

HIGHER_IS_RISKIER = frozenset(
    {
        "total_insured_value_amt",
        "dwelling_limit_amt",
        "contents_limit_amt",
        "total_finished_square_feet",
        "lifetime_claim_ct",
        "lifetime_loss_incurred_amt",
        "prior_water_claim_ct",
        "prior_nonwater_claim_ct",
        "wildfire_risk_score",
        "wildfire_score",
        "property_age_years",
        "roof_age_years",
        "no_of_stories",
    }
)

LOWER_IS_RISKIER = frozenset(
    {
        "distance_to_coast",
        "distance_to_shore",
        "distance_to_fire_hydrant_feet",
        "distance_to_fire_station_miles",
        "built_year",
        "roof_updated_year",
    }
)

SIMILARITY_FEATURES = [
    "total_insured_value_amt",
    "dwelling_limit_amt",
    "contents_limit_amt",
    "lifetime_claim_ct",
    "distance_to_coast",
    "wildfire_risk_score",
    "built_year",
]

ABSOLUTE_BLEND = 0.35
PERCENTILE_BLEND = 0.50
SIMILARITY_BLEND = 0.15

RISK_BANDS = ((0, 25, "Low"), (25, 50, "Medium"), (50, 75, "High"), (75, 100.01, "Critical"))

NULL_TOKENS = frozenset(
    {"", "NULL", "null", "None", "Unknown", "unknown", "Aop Applies", "Exclude"}
)
MONEY_RE = re.compile(r"[^\d.\-]")

NUMERIC_CLEAN = sorted(
    {f for cols in FEATURE_GROUPS.values() for f in cols}
    | {"rate_on_line", "aon_hurricane_aal_amt", "earthquake_score"}
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return None if text in NULL_TOKENS else text


def parse_money(value: Any) -> Optional[float]:
    text = clean_text(value)
    if text is None:
        return None
    cleaned = MONEY_RE.sub("", text)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_int(value: Any) -> Optional[int]:
    p = parse_money(value)
    return int(p) if p is not None else None


def parse_date(value: Any) -> Optional[date]:
    text = clean_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_yes_no(value: Any) -> Optional[bool]:
    text = clean_text(value)
    if text is None:
        return None
    low = text.lower()
    if low.startswith("yes") or low in {"y", "1", "true"}:
        return True
    if low in {"no", "n", "0", "false"}:
        return False
    return None


def risk_band(score: float) -> str:
    for low, high, label in RISK_BANDS:
        if low <= score < high:
            return label
    return "Critical"


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------


def load_results_csv(path: Path | None = None) -> pd.DataFrame:
    """Load Results.csv; assign canonical column names when the file has no header row."""
    csv_path = path or DEFAULT_CSV
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        peek = handle.readline()
        first = next(csv.reader([peek]))
        has_header = bool(first) and first[0].strip().lower() in {"policy_sk", "policy_no"}
        handle.seek(0)
        if has_header:
            df = pd.read_csv(handle)
        else:
            df = pd.read_csv(handle, header=None, names=list(RESULTS_CSV_COLUMNS))
    for col in RESULTS_CSV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df.reindex(columns=list(RESULTS_CSV_COLUMNS))


def deduplicate_submissions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    work = df.copy()
    work["_tx"] = work["transaction_seq_no"].map(parse_int)
    work["_sk"] = work["policy_sk"].map(parse_int)
    before = len(work)
    work = work.sort_values(["policy_no", "effective_dt", "_tx", "_sk"])
    work = work.drop_duplicates(subset=["policy_no", "effective_dt"], keep="last")
    removed = before - len(work)
    return work.drop(columns=["_tx", "_sk"]).reset_index(drop=True), removed


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["submission_id"] = out.apply(
        lambda r: f"{clean_text(r.get('policy_no'))}|{clean_text(r.get('effective_dt'))}",
        axis=1,
    )
    for col in NUMERIC_CLEAN:
        if col not in out.columns:
            continue
        out[f"raw_{col}"] = out[col]
        out[f"clean_{col}"] = out[col].map(parse_money)

    out["clean_occupancy_type"] = out.apply(
        lambda r: clean_text(r.get("occupancy_type"))
        or clean_text(r.get("residence_type"))
        or "Unknown",
        axis=1,
    )
    year = datetime.now().year
    if "clean_built_year" in out.columns:
        out["clean_property_age_years"] = out["clean_built_year"].map(
            lambda y: year - y if y and y > 1800 else np.nan
        )
    if "clean_roof_updated_year" in out.columns:
        out["clean_roof_age_years"] = out["clean_roof_updated_year"].map(
            lambda y: year - y if y and y > 1800 else np.nan
        )
    if "property_age_years" in out.columns:
        out["clean_property_age_years"] = out.get("clean_property_age_years", pd.Series()).fillna(
            out["property_age_years"].map(parse_money)
        )
    return out


def assign_peer_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def key(row: pd.Series) -> str:
        parts = []
        for col in PEER_GROUP_COLUMNS:
            if col == "occupancy_type":
                val = row.get("clean_occupancy_type") or clean_text(row.get("occupancy_type"))
            else:
                val = clean_text(row.get(col))
            parts.append(val or "Unknown")
        return " / ".join(parts)

    out["peer_group"] = out.apply(key, axis=1)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _scale(value: float | None, cap: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    return float(np.clip(value / cap * 100.0, 0, 100))


def absolute_risk_score(row: pd.Series) -> float:
    parts = [
        _scale(row.get("clean_total_insured_value_amt"), 30_000_000),
        min((row.get("clean_lifetime_claim_ct") or 0) * 25, 100),
        _scale(row.get("clean_lifetime_loss_incurred_amt"), 500_000),
        _scale(row.get("clean_wildfire_risk_score") or row.get("clean_wildfire_score"), 100),
    ]
    coast = row.get("clean_distance_to_coast")
    if coast is not None and not np.isnan(coast):
        parts.append(float(np.clip((50 - coast) / 50 * 100, 0, 100)))
    return round(float(np.mean(parts)), 2)


def percentile_rank(value: float, population: np.ndarray, *, higher_is_riskier: bool = True) -> float:
    if population.size <= 1 or np.isnan(value):
        return 50.0
    less = np.sum(population < value)
    equal = np.sum(population == value)
    pct = (less + 0.5 * equal) / (population.size - 1) * 100.0
    if not higher_is_riskier:
        pct = 100.0 - pct
    return round(float(np.clip(pct, 0, 100)), 2)


def _direction(feature: str) -> bool:
    if feature in LOWER_IS_RISKIER:
        return False
    return feature in HIGHER_IS_RISKIER or True


def compute_percentiles(df: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, pd.Series]:
    subset = df if mask is None else df.loc[mask]
    result: dict[str, pd.Series] = {}
    features = [f for cols in FEATURE_GROUPS.values() for f in cols]
    for feature in features:
        col = f"clean_{feature}"
        if col not in df.columns:
            continue
        pop = subset[col].dropna().astype(float).values
        if pop.size == 0:
            continue
        higher = _direction(feature)
        result[feature] = df[col].map(
            lambda v, p=pop, h=higher: percentile_rank(float(v), p, higher_is_riskier=h)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
            else np.nan
        )
    return result


def component_scores(df: pd.DataFrame, percentiles: dict[str, pd.Series]) -> pd.DataFrame:
    scores: dict[str, pd.Series] = {}
    for component, features in FEATURE_GROUPS.items():
        series_list = [percentiles[f] for f in features if f in percentiles]
        if series_list:
            scores[component] = pd.concat(series_list, axis=1).mean(axis=1, skipna=True)
        else:
            scores[component] = pd.Series(50.0, index=df.index)
    return pd.DataFrame(scores)


def score_peer_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for peer, _ in out.groupby("peer_group"):
        mask = out["peer_group"] == peer
        pct = compute_percentiles(out, mask)
        comp = component_scores(out, pct)
        for col in comp.columns:
            out.loc[mask, f"peer_pct_{col}"] = comp.loc[mask, col]
    out["peer_percentile_score"] = out[[c for c in out.columns if c.startswith("peer_pct_")]].mean(
        axis=1
    )
    return out


def similarity_context_score(df: pd.DataFrame) -> pd.Series:
    rows: dict[str, list[float]] = {}
    for feat in SIMILARITY_FEATURES:
        col = f"clean_{feat}"
        if col not in df.columns:
            continue
        vals = df[col].astype(float).fillna(df[col].median())
        std = vals.std() or 1.0
        rows[feat] = ((vals - vals.mean()) / std).tolist()
    matrix = pd.DataFrame(rows, index=df.index)
    centroid = matrix.mean(axis=0)
    dist = np.linalg.norm(matrix - centroid, axis=1)
    max_d = dist.max() or 1.0
    return pd.Series(np.clip(dist / max_d * 100, 0, 100), index=df.index)


def run_scoring(df: pd.DataFrame) -> pd.DataFrame:
    out = assign_peer_groups(df)
    out["absolute_score"] = out.apply(absolute_risk_score, axis=1)
    global_pct = compute_percentiles(out)
    global_comp = component_scores(out, global_pct)
    for col in global_comp.columns:
        out[f"global_pct_{col}"] = global_comp[col]
    out["global_percentile_score"] = global_comp.mean(axis=1)
    out = score_peer_groups(out)
    out["similarity_score"] = similarity_context_score(out)
    blend = out["global_percentile_score"] * 0.4 + out["peer_percentile_score"] * 0.6
    out["final_score"] = (
        out["absolute_score"] * ABSOLUTE_BLEND
        + blend * PERCENTILE_BLEND
        + out["similarity_score"] * SIMILARITY_BLEND
    ).round(2).clip(0, 100)
    out["risk_band"] = out["final_score"].map(risk_band)
    out["peer_group_score"] = (
        out["absolute_score"] * 0.2 + out["peer_percentile_score"] * 0.65 + out["similarity_score"] * 0.15
    ).round(2).clip(0, 100)
    return out


def load_and_score(path: Path | None = None) -> pd.DataFrame:
    raw = load_results_csv(path)
    deduped, _ = deduplicate_submissions(raw)
    cleaned = clean_dataframe(deduped)
    return run_scoring(cleaned)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "total_insured_value_amt": "Total Insured Value",
    "dwelling_limit_amt": "Dwelling Limit",
    "lifetime_claim_ct": "Lifetime Claim Count",
    "distance_to_coast": "Distance to Coast",
    "wildfire_risk_score": "Wildfire Risk Score",
}


def _percentile_at(percentiles: dict[str, pd.Series], feature: str, row_index) -> float | None:
    """Safely read a percentile for one row; missing features return None."""
    series = percentiles.get(feature)
    if series is None or series.empty:
        return None
    try:
        value = series.loc[row_index]
    except (KeyError, TypeError):
        return None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def build_explanations(df: pd.DataFrame, submission_id: str, filter_mask: pd.Series | None = None):
    matches = df.loc[df["submission_id"] == submission_id]
    if matches.empty:
        return []

    row = matches.iloc[0]
    row_index = matches.index[0]

    global_pct = compute_percentiles(df)
    peer_mask = df["peer_group"] == row["peer_group"]
    peer_pct = compute_percentiles(df, peer_mask)
    filtered_pct = compute_percentiles(df, filter_mask) if filter_mask is not None else {}

    rows = []
    all_features = sorted(set(global_pct) | set(peer_pct) | set(filtered_pct))
    for feature in all_features:
        component = next((c for c, fs in FEATURE_GROUPS.items() if feature in fs), "other")
        weight = COMPONENT_WEIGHTS.get(component, 0)
        g = _percentile_at(global_pct, feature, row_index)
        p = _percentile_at(peer_pct, feature, row_index)
        f = _percentile_at(filtered_pct, feature, row_index)
        rows.append(
            {
                "variable": FEATURE_LABELS.get(feature, feature),
                "raw_value": row.get(f"raw_{feature}", row.get(feature)),
                "clean_value": row.get(f"clean_{feature}"),
                "peer_group": row["peer_group"],
                "global_percentile": g,
                "peer_percentile": p,
                "filtered_percentile": f,
                "component": component,
                "component_weight": weight,
                "contribution": round(p * weight, 2) if p is not None else None,
            }
        )
    return sorted(rows, key=lambda x: x.get("contribution") or 0, reverse=True)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def nearest_neighbors(df: pd.DataFrame, submission_id: str, top_k: int = 10) -> pd.DataFrame:
    rows: dict[str, list[float]] = {}
    for feat in SIMILARITY_FEATURES:
        col = f"clean_{feat}"
        if col not in df.columns:
            continue
        vals = df[col].astype(float).fillna(0)
        std = vals.std() or 1.0
        rows[feat] = ((vals - vals.mean()) / std).tolist()
    matrix = pd.DataFrame(rows, index=df.index)
    idx = df.index[df["submission_id"] == submission_id][0]
    sims = cosine_similarity(matrix.loc[[idx]], matrix)[0]
    result = df.copy()
    result["similarity"] = sims
    return result[result["submission_id"] != submission_id].nlargest(top_k, "similarity")


def _source_columns() -> list[str]:
    return list(RESULTS_CSV_COLUMNS)


def generate_submission(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    row = {c: None for c in _source_columns()}
    row["policy_no"] = f"SIM-{uuid.uuid4().hex[:8].upper()}"
    row["effective_dt"] = date.today().isoformat()
    row["policy_status"] = "WIP"
    row["product_cd"] = "HO"
    for col in ["risk_state_cd", "program_type", "product_cd", "construction_type", "roof_covering"]:
        if col in df.columns and df[col].notna().any():
            row[col] = df[col].dropna().sample(1).iloc[0]
    for col in ["total_insured_value_amt", "dwelling_limit_amt", "lifetime_claim_ct", "built_year"]:
        clean = f"clean_{col}"
        if clean in df.columns:
            sample = df[clean].dropna()
            if len(sample):
                row[col] = int(rng.choice(sample.astype(int).values))
    row["insured_nm"] = f"Simulated {row['policy_no'][-4:]}"
    return row


def simulate_and_rescore(df: pd.DataFrame, count: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng()
    source_cols = _source_columns()
    base = df.reindex(columns=source_cols)
    new_rows = pd.DataFrame([generate_submission(df, rng) for _ in range(count)])
    combined = pd.concat([base, new_rows], ignore_index=True)
    deduped, _ = deduplicate_submissions(combined)
    return run_scoring(clean_dataframe(deduped))


# ---------------------------------------------------------------------------
# Notebook path setup
# ---------------------------------------------------------------------------


def setup_path() -> Path:
    """Add the notebooks folder to sys.path so `import notebook_lib` works."""
    cwd = Path.cwd()
    if (cwd / "notebook_lib.py").exists():
        nb_dir = cwd
    elif (cwd / "notebooks" / "notebook_lib.py").exists():
        nb_dir = cwd / "notebooks"
    else:
        nb_dir = cwd.parent / "notebooks"
    if str(nb_dir) not in sys.path:
        sys.path.insert(0, str(nb_dir))
    return nb_dir


# ---------------------------------------------------------------------------
# ML model comparison
# ---------------------------------------------------------------------------

ML_NUMERIC_FEATURES = [
    "total_insured_value_amt",
    "dwelling_limit_amt",
    "contents_limit_amt",
    "total_finished_square_feet",
    "lifetime_claim_ct",
    "lifetime_loss_incurred_amt",
    "prior_water_claim_ct",
    "prior_nonwater_claim_ct",
    "distance_to_coast",
    "distance_to_shore",
    "wildfire_risk_score",
    "wildfire_score",
    "property_age_years",
    "roof_age_years",
    "no_of_stories",
    "distance_to_fire_hydrant_feet",
    "distance_to_fire_station_miles",
]

ML_CATEGORICAL_FEATURES = [
    "risk_state_cd",
    "program_type",
    "construction_type",
    "roof_covering",
    "fire_protection",
]


def build_ml_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric clean_* columns plus one-hot categoricals for sklearn models."""
    parts: list[pd.DataFrame] = []
    for feat in ML_NUMERIC_FEATURES:
        col = f"clean_{feat}"
        if col in df.columns:
            parts.append(df[[col]].rename(columns={col: feat}))
    for cat in ML_CATEGORICAL_FEATURES:
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat].fillna("Unknown").astype(str), prefix=cat)
            parts.append(dummies)
    if not parts:
        raise ValueError("No ML features available in dataframe.")
    return pd.concat(parts, axis=1).fillna(0.0)


def compute_percentiles_from_reference(
    target_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Percentiles for target rows using reference population only (holdout-safe)."""
    result: dict[str, pd.Series] = {}
    features = [f for cols in FEATURE_GROUPS.values() for f in cols]
    for feature in features:
        col = f"clean_{feature}"
        if col not in target_df.columns or col not in reference_df.columns:
            continue
        pop = reference_df[col].dropna().astype(float).values
        if pop.size == 0:
            continue
        higher = _direction(feature)
        result[feature] = target_df[col].map(
            lambda v, p=pop, h=higher: percentile_rank(float(v), p, higher_is_riskier=h)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
            else np.nan
        )
    return result


def peer_holdout_scores(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Peer-group score on test rows; percentile reference = train set only."""
    test = assign_peer_groups(test.copy())
    test["absolute_score"] = test.apply(absolute_risk_score, axis=1)

    global_pct = compute_percentiles_from_reference(test, train)
    global_comp = component_scores(test, global_pct)
    global_line = global_comp.mean(axis=1)

    peer_lines: list[pd.Series] = []
    for peer in test["peer_group"].unique():
        mask = test["peer_group"] == peer
        train_ref = train[train["peer_group"] == peer]
        if len(train_ref) < 5:
            train_ref = train
        p_pct = compute_percentiles_from_reference(test.loc[mask], train_ref)
        p_comp = component_scores(test.loc[mask], p_pct)
        peer_lines.append(p_comp.mean(axis=1))

    peer_line = pd.concat(peer_lines).sort_index()
    blend = global_line * 0.4 + peer_line * 0.6
    return (
        test["absolute_score"] * ABSOLUTE_BLEND
        + blend * PERCENTILE_BLEND
        + similarity_context_score(test) * SIMILARITY_BLEND
    ).round(2).clip(0, 100)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 3)}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    if len(xr) < 2 or np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return round(float(np.corrcoef(xr, yr)[0, 1]), 3)


def run_model_comparison(
    df: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Compare peer-group holdout scoring vs Ridge, ElasticNet, Random Forest, KNN, etc.

    Regression target: holdout peer score (percentiles from train only).
    Classification target: high_priority = score >= 50 (Medium+ band).
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = build_ml_feature_matrix(df)
    idx = X.index

    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)
    train_df = df.loc[idx_train]
    test_df = df.loc[idx_test]

    y_test = peer_holdout_scores(train_df, test_df)
    y_train = peer_holdout_scores(train_df, train_df)
    y_test_vals = y_test.values
    y_train_vals = y_train.values

    claim_proxy = test_df["clean_lifetime_claim_ct"].fillna(0).values
    y_clf_test = (y_test >= 50).astype(int).values
    y_clf_train = (y_train >= 50).astype(int).values

    X_train = X.loc[idx_train]
    X_test = X.loc[idx_test]

    regressors = {
        "Ridge": Ridge(alpha=2.0),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.4, max_iter=8000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=random_state
        ),
        "KNN (k=15)": KNeighborsRegressor(n_neighbors=15, weights="distance"),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08, random_state=random_state
        ),
    }

    reg_rows: list[dict[str, Any]] = []
    peer_pred = y_test_vals
    peer_m = _regression_metrics(y_test_vals, peer_pred)
    reg_rows.append(
        {
            "model": "Peer-group (holdout)",
            "family": "Reference",
            "mae": peer_m["mae"],
            "rmse": peer_m["rmse"],
            "r2": peer_m["r2"],
            "spearman_claims": _spearman(peer_pred, claim_proxy),
        }
    )

    for name, estimator in regressors.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        pipe.fit(X_train, y_train_vals)
        pred = pipe.predict(X_test)
        m = _regression_metrics(y_test_vals, pred)
        reg_rows.append(
            {
                "model": name,
                "family": "Regression",
                **m,
                "spearman_claims": _spearman(pred, claim_proxy),
            }
        )

    regression_results = pd.DataFrame(reg_rows).sort_values("mae")

    clf_rows: list[dict[str, Any]] = []
    peer_clf_pred = (peer_pred >= 50).astype(int)
    clf_rows.append(
        {
            "model": "Peer-group (holdout)",
            "family": "Reference",
            "accuracy": round(accuracy_score(y_clf_test, peer_clf_pred), 3),
            "f1": round(f1_score(y_clf_test, peer_clf_pred, zero_division=0), 3),
            "auc": round(roc_auc_score(y_clf_test, peer_pred), 3)
            if len(np.unique(y_clf_test)) > 1
            else float("nan"),
        }
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=random_state
        ),
        "KNN (k=15)": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }

    for name, estimator in classifiers.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        pipe.fit(X_train, y_clf_train)
        pred = pipe.predict(X_test)
        proba = (
            pipe.predict_proba(X_test)[:, 1]
            if hasattr(pipe[-1], "predict_proba")
            else pred.astype(float)
        )
        auc_val = roc_auc_score(y_clf_test, proba) if len(np.unique(y_clf_test)) > 1 else float("nan")
        clf_rows.append(
            {
                "model": name,
                "family": "Classification",
                "accuracy": round(accuracy_score(y_clf_test, pred), 3),
                "f1": round(f1_score(y_clf_test, pred, zero_division=0), 3),
                "auc": round(auc_val, 3),
            }
        )

    classification_results = pd.DataFrame(clf_rows).sort_values("auc", ascending=False)

    meta = {
        "train_size": len(idx_train),
        "test_size": len(idx_test),
        "regression_target": "Holdout peer-group score (0–100)",
        "classification_target": "high_priority if holdout score >= 50",
        "claim_spearman_note": "Rank alignment with lifetime_claim_ct on test set",
    }
    return regression_results, classification_results, meta


# ---------------------------------------------------------------------------
# Report export helpers
# ---------------------------------------------------------------------------

REPORTS_DIR = NOTEBOOK_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def save_report_artifact(df: pd.DataFrame, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    df.to_csv(path, index=False)
    return path


def save_run_summary(payload: dict) -> Path:
    import json

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "run_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Charts (matplotlib) — saves PNG to reports/figures/ when the cell runs
# ---------------------------------------------------------------------------

_CHART_COLORS = {
    "navy": "#1A2332",
    "blue": "#3D6B9E",
    "gray": "#6B7785",
    "low": "#1F7A4D",
    "medium": "#9A6B16",
    "high": "#C05621",
    "critical": "#B42318",
}

_BAND_COLORS = {
    "Low": _CHART_COLORS["low"],
    "Medium": _CHART_COLORS["medium"],
    "High": _CHART_COLORS["high"],
    "Critical": _CHART_COLORS["critical"],
}


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save PNG under reports/figures/ and show in the notebook."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.show()
    plt.close(fig)
    return path


def plot_tiv_distribution(df: pd.DataFrame) -> Path:
    data = df["clean_total_insured_value_amt"].dropna() / 1e6
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=35, color=_CHART_COLORS["blue"], edgecolor="white")
    ax.set_title("Total Insured Value distribution")
    ax.set_xlabel("TIV (USD millions)")
    ax.set_ylabel("Policies")
    _style_axes(ax)
    return save_figure(fig, "01_tiv_distribution")


def plot_state_counts(df: pd.DataFrame) -> Path:
    counts = df["risk_state_cd"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index.astype(str), counts.values, color=_CHART_COLORS["navy"])
    ax.set_title("Top 12 states by submission count")
    ax.set_xlabel("Policies")
    _style_axes(ax)
    return save_figure(fig, "01_state_counts")


def plot_score_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for band, color in _BAND_COLORS.items():
        subset = df.loc[df["risk_band"] == band, "final_score"]
        if len(subset):
            ax.hist(subset, bins=20, alpha=0.75, label=band, color=color, edgecolor="white")
    ax.set_title("Peer-group risk score distribution")
    ax.set_xlabel("Risk score (0–100)")
    ax.set_ylabel("Policies")
    ax.legend()
    _style_axes(ax)
    return save_figure(fig, "02_score_distribution")


def plot_risk_bands(df: pd.DataFrame) -> Path:
    order = ["Low", "Medium", "High", "Critical"]
    counts = df["risk_band"].value_counts().reindex(order).fillna(0).astype(int)
    colors = [_BAND_COLORS[b] for b in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(order, counts.values, color=colors, edgecolor="white")
    ax.set_title("Risk band counts")
    ax.set_ylabel("Policies")
    _style_axes(ax)
    return save_figure(fig, "02_risk_bands")


def plot_peer_tiv(df: pd.DataFrame, row: pd.Series) -> Path:
    peer = df.loc[df["peer_group"] == row["peer_group"], "clean_total_insured_value_amt"].dropna() / 1e6
    tiv = row["clean_total_insured_value_amt"] / 1e6
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(peer, bins=22, color=_CHART_COLORS["blue"], alpha=0.85, edgecolor="white")
    ax.axvline(tiv, color=_CHART_COLORS["critical"], linestyle="--", linewidth=2, label=row["policy_no"])
    ax.set_title(f"TIV within peer group: {row['peer_group']}")
    ax.set_xlabel("TIV (USD millions)")
    ax.set_ylabel("Policies")
    ax.legend()
    _style_axes(ax)
    return save_figure(fig, "02_peer_tiv_example")


def plot_simulation_context(df: pd.DataFrame, new_row: pd.Series) -> Path:
    scores = df["final_score"].sort_values(ascending=False).values
    rank = int(df["final_score"].rank(ascending=False, method="min")[new_row.name])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(scores) + 1), scores, color=_CHART_COLORS["gray"], linewidth=2, label="Portfolio")
    ax.scatter([rank], [new_row["final_score"]], color=_CHART_COLORS["critical"], s=80, zorder=5, label="Simulated")
    ax.set_title(f"Simulated rank #{rank} of {len(df)} (score {new_row['final_score']:.1f})")
    ax.set_xlabel("Rank (1 = highest risk)")
    ax.set_ylabel("Risk score")
    ax.legend()
    _style_axes(ax)
    return save_figure(fig, "03_simulation_rank")


def plot_sensitivity(sensitivity_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(sensitivity_df["scenario"], sensitivity_df["mean_score_delta"], color=_CHART_COLORS["blue"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_title("Portfolio mean score shift under stress scenarios")
    ax.set_ylabel("Δ mean score (pts)")
    plt.xticks(rotation=15, ha="right")
    _style_axes(ax)
    return save_figure(fig, "04_sensitivity")


def plot_regression_mae(reg_df: pd.DataFrame) -> Path:
    plot_df = reg_df.sort_values("mae")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["model"], plot_df["mae"], color=_CHART_COLORS["blue"])
    ax.set_title("Regression MAE vs holdout peer-group score (lower is better)")
    ax.set_ylabel("MAE (points)")
    plt.xticks(rotation=20, ha="right")
    _style_axes(ax)
    return save_figure(fig, "04_regression_mae")


def plot_regression_scatter(reg_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for family in reg_df["family"].unique():
        part = reg_df[reg_df["family"] == family]
        ax.scatter(part["r2"], part["spearman_claims"], label=family, s=80)
        for _, r in part.iterrows():
            ax.annotate(r["model"], (r["r2"], r["spearman_claims"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#ccc", linestyle=":")
    ax.axvline(0, color="#ccc", linestyle=":")
    ax.set_title("R² vs claim-count rank alignment (Spearman)")
    ax.set_xlabel("R² vs holdout peer score")
    ax.set_ylabel("Spearman vs lifetime claims")
    ax.legend()
    _style_axes(ax)
    return save_figure(fig, "04_regression_scatter")


def plot_classification_auc(clf_df: pd.DataFrame) -> Path:
    plot_df = clf_df.sort_values("auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["model"], plot_df["auc"], color=_CHART_COLORS["navy"])
    ax.set_xlim(0, 1.05)
    ax.set_title("Classification AUC — high-priority triage (score ≥ 50)")
    ax.set_xlabel("AUC")
    _style_axes(ax)
    return save_figure(fig, "04_classification_auc")


def plot_final_comparison(summary_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mae_df = summary_df.dropna(subset=["mae"]).sort_values("mae")
    auc_df = summary_df.dropna(subset=["auc"]).sort_values("auc", ascending=False)
    axes[0].bar(mae_df["model"], mae_df["mae"], color=_CHART_COLORS["blue"])
    axes[0].set_title("Regression MAE ↓")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(auc_df["model"], auc_df["auc"], color=_CHART_COLORS["navy"])
    axes[1].set_title("Classification AUC ↑")
    axes[1].tick_params(axis="x", rotation=25)
    for ax in axes:
        _style_axes(ax)
    fig.suptitle("Final model comparison")
    fig.tight_layout()
    return save_figure(fig, "04_final_comparison")

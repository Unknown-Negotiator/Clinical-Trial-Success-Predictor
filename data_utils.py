"""
Utilities for loading validation data and deriving lightweight priors/penalties.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = BASE_DIR / "data" / "validation"


@dataclass
class EndpointRecord:
    abstract_id: int
    endpoint_id: int
    data: pd.Series
    has_label: bool


def _load_csv(name: str) -> pd.DataFrame:
    path = VALIDATION_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_validation_tables() -> Dict[str, pd.DataFrame]:
    """Load all validation CSVs."""
    train = _load_csv("train.csv")
    features = _load_csv("train_features.csv")
    labels = _load_csv("train_labels.csv")
    abstracts = _load_csv("abstracts.csv")
    return {
        "train": train,
        "features": features,
        "labels": labels,
        "abstracts": abstracts,
    }


def build_endpoint_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join endpoint rows with abstract-level features and labels (if present)."""
    train = tables["train"]
    features = tables["features"]
    labels = tables["labels"]
    abstracts = tables["abstracts"]

    merged = train.merge(features, on="abstract_id", how="left", suffixes=("", "_feat"))
    merged = merged.merge(labels[["endpoint_id", "endpoint_criterion_met"]], on="endpoint_id", how="left")
    merged = merged.merge(abstracts[["abstract_id", "title", "abstract_html", "trial_id"]], on="abstract_id", how="left")
    merged["endpoint_criterion_met"] = merged["endpoint_criterion_met"].astype("boolean")
    return merged


def get_endpoint(merged: pd.DataFrame, abstract_id: int, endpoint_id: int) -> EndpointRecord:
    match = merged[(merged["abstract_id"] == abstract_id) & (merged["endpoint_id"] == endpoint_id)]
    if match.empty:
        raise KeyError(f"No endpoint for abstract_id={abstract_id}, endpoint_id={endpoint_id}")
    row = match.iloc[0]
    return EndpointRecord(
        abstract_id=int(row["abstract_id"]),
        endpoint_id=int(row["endpoint_id"]),
        data=row,
        has_label=pd.notnull(row.get("endpoint_criterion_met")),
    )


def sample_random_endpoint(merged: pd.DataFrame, seed: Optional[int] = None) -> EndpointRecord:
    rng = random.Random(seed)
    idx = rng.randrange(len(merged))
    row = merged.iloc[idx]
    return EndpointRecord(
        abstract_id=int(row["abstract_id"]),
        endpoint_id=int(row["endpoint_id"]),
        data=row,
        has_label=pd.notnull(row.get("endpoint_criterion_met")),
    )


def format_endpoint_context(ep: EndpointRecord) -> str:
    """Create a concise text context for prompting."""
    d = ep.data
    parts = [
        f"Endpoint name: {d.get('endpoint_name')}",
        f"Endpoint type: {d.get('endpoint_type')}",
        f"Decision rule: {d.get('endpoint_decision_rule')}",
        f"Abstract title: {d.get('title')}",
        f"Indication: {d.get('indication')}",
        f"Phase: {d.get('phase_number')}",
        f"Trial design/comparator: {d.get('trial_design_and_comparator')}",
        f"Study size (reported): {d.get('study_size')}",
        f"Population: {d.get('study_population')}",
        f"Treatment arm: {d.get('treatment_arm')}",
        f"Control arm: {d.get('control_arm')}",
        f"Investigational product: {d.get('investigational_product')}",
        f"Mechanism/target: {d.get('mechanism_of_action')} | {d.get('target')}",
        f"Route: {d.get('route_of_administration')}",
        f"Line of therapy: {d.get('line_of_therapy')}",
        f"Regimen type: {d.get('regimen_type')}",
        f"Countries of sites: {d.get('countries_of_sites')}",
        f"Trial ID: {d.get('trial_id')}",
    ]
    abstract_txt = d.get("abstract_html")
    if isinstance(abstract_txt, str) and abstract_txt.strip():
        parts.append(f"Abstract (HTML): {abstract_txt}")
    return "\n".join([p for p in parts if p and p != "nan"])


# ---- Priors and penalties ----

def compute_base_rates(merged: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Compute Laplace-smoothed base rates by (phase_number, endpoint_type)."""
    rates: Dict[Tuple[str, str], float] = {}
    df = merged.dropna(subset=["endpoint_criterion_met"])
    if df.empty:
        return {}
    global_rate = (df["endpoint_criterion_met"].sum() + 1) / (len(df) + 2)
    rates[("GLOBAL", "GLOBAL")] = float(global_rate)
    df["phase_bucket"] = df["phase_number"].apply(_phase_bucket)
    df["endpoint_type_bucket"] = df["endpoint_type"].fillna("unknown").str.lower()

    grouped = df.groupby(["phase_bucket", "endpoint_type_bucket"])
    for (phase_b, etype), sub in grouped:
        rate = (sub["endpoint_criterion_met"].sum() + 1) / (len(sub) + 2)
        rates[(phase_b, etype)] = float(rate)
    return rates


def _phase_bucket(val) -> str:
    try:
        if pd.isna(val):
            return "unknown"
        v = float(val)
        if v >= 3:
            return "phase_3_plus"
        if v >= 2:
            return "phase_2"
        return "phase_1"
    except Exception:
        return "unknown"


def lookup_base_rate(rates: Dict[Tuple[str, str], float], row: pd.Series) -> Tuple[float, str]:
    etype = str(row.get("endpoint_type", "unknown")).lower()
    phase_b = _phase_bucket(row.get("phase_number"))
    candidates = [
        (phase_b, etype),
        (phase_b, "unknown"),
        ("unknown", etype),
        ("GLOBAL", "GLOBAL"),
    ]
    for key in candidates:
        if key in rates:
            rate = rates[key]
            return rate, f"Base rate from {key[0]} / {key[1]} (Laplace-smoothed)"
    return 0.5, "Fallback base rate 0.5 (insufficient data)"


def penalty_rules(row: pd.Series) -> List[Dict[str, object]]:
    penalties: List[Dict[str, object]] = []
    endpoint_type = str(row.get("endpoint_type", "")).lower()
    size_val = row.get("study_size")
    design = str(row.get("trial_design_and_comparator", "")).lower()
    pop = str(row.get("study_population", "")).lower()

    def add_penalty(name: str, factor: float, reason: str, flag: str = "red") -> None:
        penalties.append({"name": name, "factor": factor, "reason": reason, "flag": flag})

    is_survival = any(tok in endpoint_type for tok in ["pfs", "efs", "os", "survival"])
    if is_survival:
        try:
            if float(size_val) < 120:
                add_penalty(
                    "small_survival_n",
                    0.85,
                    "Survival endpoint with limited N; higher risk of underpowered event count.",
                    "red",
                )
        except Exception:
            pass
        if "single_arm" in design or ("randomized" not in design and "control" not in design):
            add_penalty(
                "single_arm_survival",
                0.9,
                "Survival endpoint without clear control arm.",
                "yellow",
            )

    if any(term in pop for term in ["refractory", "relapsed", "third-line", "later line"]):
        add_penalty("refractory_population", 0.9, "Refractory/relapsed population often has lower success.", "yellow")

    return penalties


APPROVED_BIOMARKERS = {
    "egfr",
    "alk",
    "her2",
    "braf",
    "kras",
    "pdl1",
    "pd-l1",
    "pd1",
    "pd-1",
    "ctla4",
    "ctla-4",
    "vegf",
}


def biomarker_flag(row: pd.Series) -> Optional[Dict[str, str]]:
    target = str(row.get("target", "")).lower()
    if not target or target == "nan":
        return None
    for token in re.split(r"[\\s,;/]+", target):
        if token in APPROVED_BIOMARKERS:
            return {"level": "green", "reason": f"Target {token} has established approvals; lowers risk."}
    return {"level": "yellow", "reason": f"Target {target} not recognized as established biomarker; uncertain risk."}


def blend_probability(llm_prob: float, base_prob: float, penalties: List[Dict[str, object]]) -> Tuple[float, List[Dict[str, object]]]:
    llm_prob = float(np.clip(llm_prob, 0.0, 1.0))
    base_prob = float(np.clip(base_prob, 0.0, 1.0))
    blended = 0.6 * llm_prob + 0.4 * base_prob
    applied: List[Dict[str, object]] = []
    for p in penalties:
        blended *= float(p["factor"])
        applied.append(p)
    blended = float(np.clip(blended, 0.0, 1.0))
    return blended, applied

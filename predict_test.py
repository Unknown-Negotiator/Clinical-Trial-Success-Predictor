"""
Generate agent predictions for the test endpoints and write a submission file.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from agent_utils import AgentError, build_client, run_agent
from data_utils import (
    EndpointRecord,
    VALIDATION_DIR,
    biomarker_flag,
    blend_probability,
    build_endpoint_table,
    compute_base_rates,
    format_endpoint_context,
    load_validation_tables,
    lookup_base_rate,
    penalty_rules,
)
from retrieval_utils import Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent predictions on test endpoints and create a submission CSV.")
    parser.add_argument("--limit", type=int, default=-1, help="Number of test endpoints to score (-1 = all).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed when subsetting with --limit.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on blended probability.")
    parser.add_argument("--output", type=Path, default=Path("data/results/sample_submission.csv"), help="Path to submission CSV.")
    parser.add_argument("--details", type=Path, default=None, help="Optional JSONL with detailed per-endpoint outputs.")
    parser.add_argument("--model", type=str, default=None, help="Override model name (otherwise uses OPENAI_MODEL or default).")
    return parser.parse_args()


def load_test_table() -> pd.DataFrame:
    test = pd.read_csv(VALIDATION_DIR / "test.csv")
    features = pd.read_csv(VALIDATION_DIR / "test_features.csv")
    abstracts = pd.read_csv(VALIDATION_DIR / "abstracts.csv")
    merged = test.merge(features, on="abstract_id", how="left", suffixes=("", "_feat"))
    merged = merged.merge(abstracts[["abstract_id", "title", "abstract_html", "trial_id"]], on="abstract_id", how="left")
    merged["endpoint_criterion_met"] = pd.NA
    return merged


def build_query(row) -> str:
    parts = [
        row.get("endpoint_name"),
        row.get("endpoint_type"),
        row.get("indication"),
        row.get("investigational_product"),
        row.get("mechanism_of_action"),
    ]
    return " ".join([str(p) for p in parts if p and str(p) != "nan"])


def make_endpoint_record(row) -> EndpointRecord:
    return EndpointRecord(
        abstract_id=int(row["abstract_id"]),
        endpoint_id=int(row["endpoint_id"]),
        data=row,
        has_label=False,
    )


def main() -> None:
    args = parse_args()

    # Base rates from the labeled training set
    train_tables = load_validation_tables()
    merged_train = build_endpoint_table(train_tables)
    base_rates = compute_base_rates(merged_train)

    test_df = load_test_table()

    retriever = Retriever(Path("data") / "rag" / "rag_index")
    client = build_client()

    idxs = list(range(len(test_df)))
    rng = random.Random(args.seed)
    rng.shuffle(idxs)
    if args.limit and args.limit > 0:
        idxs = idxs[: args.limit]

    submission_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    errors = 0

    for i, idx in enumerate(idxs, 1):
        row = test_df.loc[idx]
        ep = make_endpoint_record(row)
        base_prob, base_note = lookup_base_rate(base_rates, row)
        penalties = penalty_rules(row)
        biomarker = biomarker_flag(row)
        query = build_query(row)
        retrieved = retriever.search(query, k=6)
        context = format_endpoint_context(ep)

        try:
            model_res, _raw = run_agent(
                client=client,
                trial_context=context,
                base_rate=base_prob,
                base_note=base_note,
                penalties=penalties,
                biomarker=biomarker,
                retrieved=retrieved,
                model=args.model,
            )
        except AgentError as e:
            errors += 1
            print(f"[{i}/{len(idxs)}] abstract {ep.abstract_id} endpoint {ep.endpoint_id} -> error: {e}", file=sys.stderr)
            continue

        llm_prob = float(model_res.get("llm_probability", 0.5))
        blended, applied = blend_probability(llm_prob, base_prob, penalties)
        pred = 1 if blended >= args.threshold else 0

        submission_rows.append({"endpoint_id": int(ep.endpoint_id), "endpoint_criterion_met": pred})

        if args.details:
            detail_rows.append(
                {
                    "abstract_id": int(ep.abstract_id),
                    "endpoint_id": int(ep.endpoint_id),
                    "prediction": pred,
                    "blended_prob": blended,
                    "llm_prob": llm_prob,
                    "base_prob": base_prob,
                    "base_note": base_note,
                    "penalties": applied,
                    "biomarker": biomarker,
                    "flags": model_res.get("flags", {}),
                    "rationale": model_res.get("rationale", ""),
                    "citations": model_res.get("citations", []),
                }
            )

        if i % 5 == 0 or i == len(idxs):
            print(f"[{i}/{len(idxs)}] prob={blended:.3f} pred={pred}")

    # Write outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(submission_rows).to_csv(args.output, index=False)
    print(f"Wrote submission to {args.output} | errors={errors}")

    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        with args.details.open("w") as f:
            for row in detail_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote details to {args.details}")


if __name__ == "__main__":
    main()

"""
Batch evaluation script for the Streamlit agent pipeline.

Runs the RAG + prior + penalty + LLM flow over labeled endpoints and reports metrics.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from agent_utils import AgentError, build_client, run_agent
from data_utils import (
    EndpointRecord,
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
    parser = argparse.ArgumentParser(description="Evaluate agent on labeled endpoints.")
    parser.add_argument("--limit", type=int, default=20, help="Number of endpoints to evaluate (default 20). Use -1 for all.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on blended probability.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL log file for per-endpoint results.")
    parser.add_argument("--model", type=str, default=None, help="Override model name (otherwise uses OPENAI_MODEL env or default).")
    return parser.parse_args()


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
    has_label = row.get("endpoint_criterion_met")
    return EndpointRecord(
        abstract_id=int(row["abstract_id"]),
        endpoint_id=int(row["endpoint_id"]),
        data=row,
        has_label=bool(has_label) if has_label is not None else False,
    )


def evaluate(limit: int, seed: int, threshold: float, output: Path | None, model_override: str | None) -> None:
    tables = load_validation_tables()
    merged = build_endpoint_table(tables)
    labeled = merged.dropna(subset=["endpoint_criterion_met"]).reset_index(drop=True)
    base_rates = compute_base_rates(merged)

    retriever = Retriever(Path("data") / "rag" / "rag_index")
    client = build_client()

    idxs = list(range(len(labeled)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if limit and limit > 0:
        idxs = idxs[:limit]

    y_true: List[int] = []
    y_pred: List[int] = []
    probs: List[float] = []
    logs: List[Dict[str, object]] = []
    errors = 0

    start_time = time.time()
    for i, idx in enumerate(idxs, 1):
        row = labeled.loc[idx]
        ep = make_endpoint_record(row)
        base_prob, base_note = lookup_base_rate(base_rates, row)
        penalties = penalty_rules(row)
        biomark = biomarker_flag(row)
        query = build_query(row)
        retrieved = retriever.search(query, k=6)
        context = format_endpoint_context(ep)
        try:
            model_res, raw = run_agent(
                client=client,
                trial_context=context,
                base_rate=base_prob,
                base_note=base_note,
                penalties=penalties,
                biomarker=biomark,
                retrieved=retrieved,
                model=model_override,
            )
        except AgentError as e:
            errors += 1
            print(f"[{i}/{len(idxs)}] abstract {ep.abstract_id} endpoint {ep.endpoint_id} -> error {e}", file=sys.stderr)
            continue

        llm_prob = float(model_res.get("llm_probability", 0.5))
        blended, applied = blend_probability(llm_prob, base_prob, penalties)
        pred = 1 if blended >= threshold else 0
        truth = int(row["endpoint_criterion_met"])

        y_true.append(truth)
        y_pred.append(pred)
        probs.append(blended)

        if output:
            logs.append(
                {
                    "abstract_id": ep.abstract_id,
                    "endpoint_id": ep.endpoint_id,
                    "truth": truth,
                    "pred": pred,
                    "blended_prob": blended,
                    "llm_prob": llm_prob,
                    "base_prob": base_prob,
                    "base_note": base_note,
                    "penalties": applied,
                    "biomarker": biomark,
                    "flags": model_res.get("flags", {}),
                    "rationale": model_res.get("rationale", ""),
                    "citations": model_res.get("citations", []),
                }
            )

        if i % 5 == 0 or i == len(idxs):
            print(f"[{i}/{len(idxs)}] last prob={blended:.3f} pred={pred} truth={truth}")

    if not y_true:
        print("No examples evaluated.")
        return

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    elapsed = time.time() - start_time
    print(f"n={len(y_true)} | acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} | time={elapsed/60:.1f} min | errors={errors}")

    if output:
        with output.open("w") as f:
            for row in logs:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote logs to {output}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(limit=args.limit if args.limit != -1 else None, seed=args.seed, threshold=args.threshold, output=args.output, model_override=args.model)

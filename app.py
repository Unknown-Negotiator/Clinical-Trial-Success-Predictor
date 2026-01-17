from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import streamlit as st

from agent_utils import AgentError, build_client, run_agent
from data_utils import (
    biomarker_flag,
    blend_probability,
    build_endpoint_table,
    compute_base_rates,
    format_endpoint_context,
    EndpointRecord,
    get_endpoint,
    load_validation_tables,
    lookup_base_rate,
    penalty_rules,
    sample_random_endpoint,
)
from retrieval_utils import Retriever


st.set_page_config(page_title="Clinical Trial Endpoint Agent", layout="wide")


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    rag_dir = Path("data") / "rag" / "rag_index"
    return Retriever(rag_dir)


@st.cache_resource(show_spinner=False)
def get_client():
    return build_client()


@st.cache_data(show_spinner=False)
def load_data():
    tables = load_validation_tables()
    merged = build_endpoint_table(tables)
    base_rates = compute_base_rates(merged)
    return merged, base_rates


def _build_query(row) -> str:
    parts = [
        row.get("endpoint_name", ""),
        row.get("endpoint_type", ""),
        row.get("indication", ""),
        row.get("investigational_product", ""),
        row.get("mechanism_of_action", ""),
    ]
    return " ".join([str(p) for p in parts if p and str(p) != "nan"])


def _merge_flags(model_flags: Dict[str, List[str]], penalties: List[Dict[str, object]], biomarker: Dict[str, str] | None):
    flags = {"red": [], "yellow": [], "green": []}
    if model_flags:
        for k in flags:
            if k in model_flags and isinstance(model_flags[k], list):
                flags[k].extend([str(x) for x in model_flags[k]])
    for p in penalties:
        level = p.get("flag", "yellow")
        flags.get(level, flags["yellow"]).append(p["reason"])
    if biomarker:
        flags.get(biomarker["level"], flags["yellow"]).append(biomarker["reason"])
    return flags


def run_prediction(ep_row, base_rates):
    retriever = get_retriever()
    client = get_client()

    penalties = penalty_rules(ep_row.data)
    biomarker = biomarker_flag(ep_row.data)
    base_prob, base_note = lookup_base_rate(base_rates, ep_row.data)
    query = _build_query(ep_row.data)
    retrieved = retriever.search(query, k=6)
    trial_context = format_endpoint_context(ep_row)

    model_result, raw_content = run_agent(
        client=client,
        trial_context=trial_context,
        base_rate=base_prob,
        base_note=base_note,
        penalties=penalties,
        biomarker=biomarker,
        retrieved=retrieved,
    )

    llm_prob = float(model_result.get("llm_probability", 0.5))
    blended_prob, applied = blend_probability(llm_prob, base_prob, penalties)
    flags = _merge_flags(model_result.get("flags", {}), applied, biomarker)
    final_pred = "yes" if blended_prob >= 0.5 else "no"

    return {
        "model_result": model_result,
        "raw_content": raw_content,
        "retrieved": retrieved,
        "base_prob": base_prob,
        "base_note": base_note,
        "penalties": applied,
        "biomarker": biomarker,
        "blended_prob": blended_prob,
        "final_prediction": final_pred,
        "flags": flags,
    }


def _render_flags(flags: Dict[str, List[str]]):
    colors = {"red": "#ff6b6b", "yellow": "#f4d03f", "green": "#58d68d"}
    for level in ["red", "yellow", "green"]:
        items = flags.get(level, [])
        if not items:
            continue
        st.markdown(f"**{level.upper()} flags:**")
        for item in items:
            st.markdown(f"- <span style='color:{colors[level]};'>■</span> {item}", unsafe_allow_html=True)


def main():
    st.title("Clinical Trial Endpoint Agent")
    st.caption("Single-endpoint assessor with RAG evidence, priors, and flagging.")

    merged, base_rates = load_data()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Select validation endpoint")
        abstract_ids = sorted(merged["abstract_id"].unique())
        selected_abs = st.selectbox("Abstract ID", abstract_ids)
        subset = merged[merged["abstract_id"] == selected_abs]
        endpoint_options = [
            f"{int(r.endpoint_id)} — {r.endpoint_name} ({r.endpoint_type})"
            for r in subset[["endpoint_id", "endpoint_name", "endpoint_type"]].itertuples(index=False)
        ]
        chosen = st.selectbox("Endpoint", endpoint_options)
        endpoint_id = int(chosen.split(" — ")[0])
        if st.button("Run prediction", type="primary"):
            ep = get_endpoint(merged, selected_abs, endpoint_id)
            with st.spinner("Running agent..."):
                try:
                    result = run_prediction(ep, base_rates)
                except AgentError as e:
                    st.error(str(e))
                except Exception as e:  # pragma: no cover - UI safety
                    st.error(f"Unexpected error: {e}")
                else:
                    render_result(ep, result)

    with col2:
        st.subheader("Quick demo")
        if st.button("Random labeled endpoint"):
            ep = sample_random_endpoint(merged)
            st.write(f"Picked abstract {ep.abstract_id}, endpoint {ep.endpoint_id}")
            with st.spinner("Running agent..."):
                try:
                    result = run_prediction(ep, base_rates)
                except AgentError as e:
                    st.error(str(e))
                except Exception as e:  # pragma: no cover - UI safety
                    st.error(f"Unexpected error: {e}")
                else:
                    render_result(ep, result)
        st.subheader("Custom endpoint (JSON)")
        st.caption(
            "Provide fields like endpoint_name, endpoint_type, indication, phase_number, study_size, "
            "trial_design_and_comparator, investigational_product, mechanism_of_action, target, study_population."
        )
        user_json = st.text_area("Custom endpoint JSON", height=220, placeholder='{"endpoint_name": "...", "endpoint_type": "...", "phase_number": 2, "study_size": 120, "indication": "..."}')
        if st.button("Run custom JSON"):
            try:
                payload = json.loads(user_json)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
            else:
                ep = _make_custom_endpoint(payload)
                with st.spinner("Running agent..."):
                    try:
                        result = run_prediction(ep, base_rates)
                    except AgentError as e:
                        st.error(str(e))
                    except Exception as e:  # pragma: no cover - UI safety
                        st.error(f"Unexpected error: {e}")
                    else:
                        render_result(ep, result)


def _make_custom_endpoint(payload: Dict[str, object]):
    import pandas as pd

    fields = [
        "abstract_id",
        "endpoint_id",
        "endpoint_name",
        "endpoint_type",
        "endpoint_decision_rule",
        "indication",
        "phase_number",
        "trial_design_and_comparator",
        "study_size",
        "study_population",
        "treatment_arm",
        "control_arm",
        "investigational_product",
        "mechanism_of_action",
        "target",
        "route_of_administration",
        "line_of_therapy",
        "regimen_type",
        "countries_of_sites",
        "title",
        "trial_id",
    ]
    data = {k: payload.get(k) for k in fields}
    data["abstract_id"] = data.get("abstract_id") or 999999
    data["endpoint_id"] = data.get("endpoint_id") or 1
    row = pd.Series(data)
    from data_utils import EndpointRecord

    return EndpointRecord(abstract_id=int(row["abstract_id"]), endpoint_id=int(row["endpoint_id"]), data=row, has_label=False)


def render_result(ep, result):
    st.divider()
    st.markdown(f"### Endpoint {ep.endpoint_id} — {ep.data.get('endpoint_name')}")
    st.markdown(f"**Decision:** {result['final_prediction'].upper()}  |  Probability: {result['blended_prob']:.2f}")
    st.markdown(f"Base rate: {result['base_prob']:.2f} ({result['base_note']})")
    if result["penalties"]:
        st.markdown("Applied penalties:")
        for p in result["penalties"]:
            st.markdown(f"- {p['name']} (x{p['factor']}) — {p['reason']}")
    _render_flags(result["flags"])
    st.markdown("**Rationale:**")
    st.write(result["model_result"].get("rationale", ""))
    st.markdown("**Citations:**")
    citations = result["model_result"].get("citations", [])
    if citations:
        for c in citations:
            st.markdown(f"- [{c.get('title', '')}]({c.get('url', '')})")
    else:
        st.write("No citations provided.")
    with st.expander("Retrieved evidence"):
        for hit in result["retrieved"]:
            st.markdown(f"**{hit['title']}** ({hit.get('source')})")
            st.write(hit["text"])
            if hit.get("url"):
                st.write(hit["url"])
            st.markdown("---")
    with st.expander("Raw LLM content"):
        st.code(result["raw_content"])


if __name__ == "__main__":
    try:
        main()
    except AgentError as e:
        st.error(str(e))

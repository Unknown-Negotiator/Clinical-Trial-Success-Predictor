"""
LLM agent runner: builds prompts, calls OpenAI-compatible API, and parses structured output.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from openai import OpenAI


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class AgentError(Exception):
    pass


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AgentError("OPENAI_API_KEY is not set.")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _format_retrieved(hits: List[Dict[str, str]]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(
            f"[{i}] Title: {h.get('title')} | Source: {h.get('source')} | URL: {h.get('url')}\nExcerpt: {h.get('text')}"
        )
    return "\n\n".join(lines)


def build_prompt(trial_context: str, base_rate: float, base_note: str, penalties: List[Dict[str, object]], biomarker: Dict[str, str] | None, retrieved: List[Dict[str, str]]) -> str:
    penalty_lines = [f"- {p['name']}: x{p['factor']} ({p['reason']})" for p in penalties] or ["- none"]
    biomarker_line = f"{biomarker['level'].upper()} - {biomarker['reason']}" if biomarker else "Not specified"
    retrieved_block = _format_retrieved(retrieved)
    prompt = f"""
You are an oncology trial endpoint assessor. Use only the provided trial details and retrieved evidence.
Be concise, avoid speculation, and cite only from the retrieved snippets.

Trial details:
{trial_context}

Base rate prior: {base_rate:.3f} ({base_note})
Biomarker tier: {biomarker_line}
Penalty hints (apply caution when relevant):
{chr(10).join(penalty_lines)}

Retrieved evidence:
{retrieved_block or "No evidence retrieved"}

Return JSON only with keys:
{{
  "prediction": "yes" or "no",
  "llm_probability": float between 0 and 1,
  "flags": {{"red": [], "yellow": [], "green": []}},
  "rationale": "one short paragraph focusing on design, sample size, endpoint robustness",
  "citations": [{{"title": "...", "url": "..."}}]  // derived from retrieved evidence
}}
Do not add extra keys or text. If evidence is weak, lower probability and add a yellow flag.
"""
    return prompt.strip()


def run_agent(
    client: OpenAI,
    trial_context: str,
    base_rate: float,
    base_note: str,
    penalties: List[Dict[str, object]],
    biomarker: Dict[str, str] | None,
    retrieved: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
) -> Tuple[Dict[str, object], str]:
    model = model or DEFAULT_MODEL
    user_prompt = build_prompt(trial_context, base_rate, base_note, penalties, biomarker, retrieved)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "You are a cautious oncology endpoint reviewer. Always respond with strict JSON only.",
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content
    if content is None:
        raise AgentError("LLM returned empty content.")
    parsed = _parse_json_block(content)
    return parsed, content


def _parse_json_block(text: str) -> Dict[str, object]:
    def attempt_load(candidate: str) -> Dict[str, object]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Heuristic cleanup: fix single quotes, trailing commas, Python booleans
            fixed = candidate
            fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            fixed = fixed.replace("'", '"')
            return json.loads(fixed)

    try:
        return attempt_load(text)
    except json.JSONDecodeError:
        # Try to isolate the JSON portion (strip fences if present)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return attempt_load(snippet)
            except json.JSONDecodeError:
                pass
    raise AgentError("Failed to parse JSON from model response.")

"""
Qualitative analysis phase (Section 8 of TRAJECTORY_ANALYSIS_PROTOCOL.md).

Runs after protocol convergence for each scenario. Three sub-tasks:
  Rubric evaluation       — score representative cluster trajectories on 5 dimensions
  Gold standard comparison — compare representative trajectories to ideal diagnosis
  Emergent findings       — LLM-assisted open-ended pattern extraction

Entry point: run_qualitative_analysis(...)

Structured output is also serialised to qualitative.json next to final_report.txt
so that downstream scripts can aggregate across scenarios, systems, or fault types.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Rating values for rubric dimensions — kept as a plain string so weak models
# that output "medium-high" or free text don't break parsing, but callers that
# want to aggregate should normalise to {"high","medium","low","N/A"}.
@dataclass
class DimensionScore:
    rating: str    # "high" | "medium" | "low"
    rationale: str


@dataclass
class RubricScore:
    cluster_id: int
    cluster_label: str
    actionability: DimensionScore
    diagnostic_coherence: DimensionScore
    efficiency: DimensionScore
    consistency: DimensionScore
    evidence_usage: DimensionScore
    overall_comment: str


@dataclass
class GoldComparison:
    cluster_id: int
    cluster_label: str
    injected_fault: str
    gold_diagnosis: str
    match_level: str          # "correct" | "partial" | "incorrect" | "different_strategy"
    explanation: str


@dataclass
class EmergentFindings:
    failure_modes: list[str] = field(default_factory=list)
    novel_strategies: list[str] = field(default_factory=list)
    inefficiencies: list[str] = field(default_factory=list)
    candidate_metrics: list[str] = field(default_factory=list)


@dataclass
class QualitativeReport:
    scenario_number: int
    rubric_scores: list[RubricScore] = field(default_factory=list)
    gold_comparisons: list[GoldComparison] = field(default_factory=list)
    emergent_findings: Optional[EmergentFindings] = None


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class _DimOutput(BaseModel):
    rating: str      # high | medium | low
    rationale: str


class _RubricOutput(BaseModel):
    actionability: _DimOutput
    diagnostic_coherence: _DimOutput
    efficiency: _DimOutput
    consistency: _DimOutput
    evidence_usage: _DimOutput
    overall_comment: str


class _GoldOutput(BaseModel):
    match_level: str   # correct | partial | incorrect | different_strategy
    explanation: str


class _EmergentOutput(BaseModel):
    failure_modes: list[str]
    novel_strategies: list[str]
    inefficiencies: list[str]
    candidate_metrics: list[str]


# ---------------------------------------------------------------------------
# Gold label loading from SCENARIOS_MASTER.csv
# ---------------------------------------------------------------------------

_GOLD_CACHE: dict[int, dict] = {}


def _load_gold(scenario_number: int, csv_path: Optional[Path] = None) -> dict:
    """Return {injected_fault, ideal_diagnosis} for a scenario number."""
    if scenario_number in _GOLD_CACHE:
        return _GOLD_CACHE[scenario_number]

    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "SCENARIOS_MASTER.csv"

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            try:
                num = int(row[0])
            except ValueError:
                continue
            _GOLD_CACHE[num] = {
                "injected_fault":  row[6].strip() if len(row) > 6 else "",
                "ideal_diagnosis": row[9].strip() if len(row) > 9 else "",
            }

    return _GOLD_CACHE.get(scenario_number, {"injected_fault": "", "ideal_diagnosis": ""})


# ---------------------------------------------------------------------------
# Trajectory formatting
# ---------------------------------------------------------------------------

def _format_trajectory(traj: dict) -> str:
    lines = [f"outcome: {traj.get('end', '?')} | cost: {traj.get('total_cost', '?')}"]
    for i, action in enumerate(traj.get("actions", []), 1):
        intention = action.get("intention", "")
        outcome   = action.get("outcome", "")
        lines.append(f"  {i}. {intention}")
        if outcome:
            lines.append(f"     → {outcome}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Representative trajectory selection
# ---------------------------------------------------------------------------

def _representative(
    trajectories: list[dict],
    labels: list[int],
    probabilities,
    cluster_id: int,
) -> dict:
    members = [i for i, l in enumerate(labels) if l == cluster_id]
    if not members:
        return {}
    if probabilities is not None:
        try:
            best = max(members, key=lambda i: probabilities[i])
        except Exception:
            best = members[0]
    else:
        best = members[0]
    return trajectories[best]


# ---------------------------------------------------------------------------
# Rubric definition (shared between rubric evaluation and emergent findings)
# ---------------------------------------------------------------------------

_RUBRIC_DIMENSIONS_TEXT = """\
Rubric dimensions:
- actionability: are the suggested actions concrete and executable on the physical system?
- diagnostic_coherence: does the sequence of actions logically narrow down the fault location?
- efficiency: does the agent avoid redundant, repeated, or off-target actions?
- consistency: are the actions consistent with prior observations and test results?
- evidence_usage: does the agent update its belief / hypothesis after each measurement?"""


# ---------------------------------------------------------------------------
# Rubric evaluation
# ---------------------------------------------------------------------------

_RUBRIC_SYSTEM = (
    "You are an expert in AI-driven diagnosis evaluation. "
    "You will be given a diagnostic trajectory produced by an AI agent on an electrical system. "
    "Score the trajectory on 5 dimensions. For each dimension give a rating (high/medium/low) "
    "and a single concise sentence of rationale. Then give a brief overall comment."
)


def evaluate_rubric(
    trajectories: list[dict],
    labels: list[int],
    probabilities,
    cluster_labels: dict[int, str],
    model: str,
) -> list[RubricScore]:
    from openai import OpenAI
    client = OpenAI()
    cluster_ids = sorted(set(l for l in labels if l >= 0))
    scores: list[RubricScore] = []

    for cid in cluster_ids:
        traj = _representative(trajectories, labels, probabilities, cid)
        if not traj:
            continue
        text = _format_trajectory(traj)
        label = cluster_labels.get(cid, f"Cluster {cid}")

        try:
            resp = client.beta.chat.completions.parse(
                model=model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": _RUBRIC_SYSTEM},
                    {"role": "user", "content": (
                        f"{_RUBRIC_DIMENSIONS_TEXT}\n\nTrajectory:\n{text}"
                    )},
                ],
                response_format=_RubricOutput,
            )
            out: _RubricOutput = resp.choices[0].message.parsed
            scores.append(RubricScore(
                cluster_id=cid,
                cluster_label=label,
                actionability=DimensionScore(rating=out.actionability.rating, rationale=out.actionability.rationale),
                diagnostic_coherence=DimensionScore(rating=out.diagnostic_coherence.rating, rationale=out.diagnostic_coherence.rationale),
                efficiency=DimensionScore(rating=out.efficiency.rating, rationale=out.efficiency.rationale),
                consistency=DimensionScore(rating=out.consistency.rating, rationale=out.consistency.rationale),
                evidence_usage=DimensionScore(rating=out.evidence_usage.rating, rationale=out.evidence_usage.rationale),
                overall_comment=out.overall_comment,
            ))
        except Exception as e:
            na = DimensionScore(rating="N/A", rationale="")
            scores.append(RubricScore(
                cluster_id=cid,
                cluster_label=label,
                actionability=na,
                diagnostic_coherence=na,
                efficiency=na,
                consistency=na,
                evidence_usage=na,
                overall_comment=f"(rubric error: {e})",
            ))

    return scores


# ---------------------------------------------------------------------------
# Gold standard comparison
# ---------------------------------------------------------------------------

_GOLD_SYSTEM = (
    "You are evaluating an AI diagnostic agent against a gold-standard diagnosis. "
    "Classify the match as one of: correct, partial, incorrect, different_strategy. "
    "correct = same fault identified via same or equivalent steps. "
    "partial = correct fault but took unnecessary detours or missed steps. "
    "incorrect = wrong fault identified. "
    "different_strategy = correct fault but via a substantially different valid approach. "
    "Then explain in one concise sentence."
)


def compare_to_gold(
    trajectories: list[dict],
    labels: list[int],
    probabilities,
    cluster_labels: dict[int, str],
    scenario_number: int,
    model: str,
) -> list[GoldComparison]:
    from openai import OpenAI
    client = OpenAI()
    gold = _load_gold(scenario_number)
    cluster_ids = sorted(set(l for l in labels if l >= 0))
    comparisons: list[GoldComparison] = []

    for cid in cluster_ids:
        traj = _representative(trajectories, labels, probabilities, cid)
        if not traj:
            continue
        text = _format_trajectory(traj)
        label = cluster_labels.get(cid, f"Cluster {cid}")

        try:
            resp = client.beta.chat.completions.parse(
                model=model,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": _GOLD_SYSTEM},
                    {"role": "user", "content": (
                        f"Injected fault: {gold['injected_fault']}\n"
                        f"Gold diagnosis: {gold['ideal_diagnosis']}\n\n"
                        f"Actual trajectory:\n{text}"
                    )},
                ],
                response_format=_GoldOutput,
            )
            out: _GoldOutput = resp.choices[0].message.parsed
            comparisons.append(GoldComparison(
                cluster_id=cid,
                cluster_label=label,
                injected_fault=gold["injected_fault"],
                gold_diagnosis=gold["ideal_diagnosis"],
                match_level=out.match_level,
                explanation=out.explanation,
            ))
        except Exception as e:
            comparisons.append(GoldComparison(
                cluster_id=cid,
                cluster_label=label,
                injected_fault=gold["injected_fault"],
                gold_diagnosis=gold["ideal_diagnosis"],
                match_level="N/A",
                explanation=f"(gold comparison error: {e})",
            ))

    return comparisons


# ---------------------------------------------------------------------------
# Emergent findings
# ---------------------------------------------------------------------------

_EMERGENT_FIELD_NAMES = {"failure_modes", "novel_strategies", "inefficiencies", "candidate_metrics"}


def _clean_list(items: list[str]) -> list[str]:
    """Remove field-name echoes that weak models sometimes emit."""
    return [s for s in items if s.strip().lower().rstrip(":") not in _EMERGENT_FIELD_NAMES and s.strip()]


_EMERGENT_SYSTEM = (
    "You are analyzing diagnostic trajectories produced by an AI agent. "
    "Identify cross-trajectory patterns that go beyond what the standard rubric already captures. "
    "Return structured lists of short, concrete strings. "
    "Leave a list empty ([]) if you have no notable observation for that category — do not force findings."
)

_EMERGENT_USER_TEMPLATE = """\
Standard rubric (for reference — identify patterns BEYOND these dimensions):
{rubric}

Representative diagnostic trajectories (one per behavioral cluster):
{trajectories}

Identify:
- failure_modes: recurring ways the agent fails or gets stuck (beyond rubric dimensions)
- novel_strategies: approaches not anticipated by the rubric that appeared effective or interesting
- inefficiencies: common wasteful patterns not already captured by the efficiency dimension
- candidate_metrics: new measurable quantities that could extend the rubric

Return empty lists for categories where nothing notable was observed."""


def extract_emergent_findings(
    trajectories: list[dict],
    labels: list[int],
    probabilities,
    model: str,
) -> EmergentFindings:
    from openai import OpenAI
    client = OpenAI()
    cluster_ids = sorted(set(l for l in labels if l >= 0))
    if not cluster_ids:
        return EmergentFindings()

    reps = []
    for cid in cluster_ids:
        traj = _representative(trajectories, labels, probabilities, cid)
        if traj:
            reps.append(f"[Cluster {cid}]\n{_format_trajectory(traj)}")

    combined = "\n\n".join(reps)
    user_msg = _EMERGENT_USER_TEMPLATE.format(
        rubric=_RUBRIC_DIMENSIONS_TEXT,
        trajectories=combined,
    )

    try:
        resp = client.beta.chat.completions.parse(
            model=model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": _EMERGENT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=_EmergentOutput,
        )
        out: _EmergentOutput = resp.choices[0].message.parsed
        return EmergentFindings(
            failure_modes=_clean_list(out.failure_modes),
            novel_strategies=_clean_list(out.novel_strategies),
            inefficiencies=_clean_list(out.inefficiencies),
            candidate_metrics=_clean_list(out.candidate_metrics),
        )
    except Exception as e:
        return EmergentFindings(failure_modes=[f"(emergent analysis error: {e})"])


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def save_qualitative_json(report: QualitativeReport, path: Path) -> None:
    """Write the full structured report to a JSON file for downstream aggregation."""
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_qualitative_analysis(
    scenario_number: int,
    trajectories: list[dict],
    cluster_assignments_intent: list[int],
    cluster_probabilities,
    cluster_labels: dict[int, str],
    model: str = "gpt-4.1",
    output_dir: Optional[Path] = None,
) -> QualitativeReport:
    """
    Run all three qualitative sub-tasks and return a QualitativeReport.
    If output_dir is given, also writes qualitative.json there.
    Safe to call even if there are no clusters (returns empty report).
    """
    labels = cluster_assignments_intent or []

    # If HDBSCAN assigned everything as noise, treat all as one implicit cluster.
    if not any(l >= 0 for l in labels) and trajectories:
        labels = [0] * len(trajectories)
        cluster_labels = cluster_labels or {0: "all trajectories (no clusters found)"}
        cluster_probabilities = None

    has_clusters = bool(trajectories)

    rubric: list[RubricScore] = []
    gold: list[GoldComparison] = []
    emergent: Optional[EmergentFindings] = None

    if has_clusters:
        rubric   = evaluate_rubric(trajectories, labels, cluster_probabilities, cluster_labels, model)
        gold     = compare_to_gold(trajectories, labels, cluster_probabilities, cluster_labels, scenario_number, model)
        emergent = extract_emergent_findings(trajectories, labels, cluster_probabilities, model)

    report = QualitativeReport(
        scenario_number=scenario_number,
        rubric_scores=rubric,
        gold_comparisons=gold,
        emergent_findings=emergent,
    )

    if output_dir is not None:
        try:
            save_qualitative_json(report, output_dir / "qualitative.json")
        except Exception:
            pass

    return report
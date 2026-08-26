"""awevolve — point an agent at a file and a command that scores it.

An evolutionary search whose variation operator is an AGENT rather than a single
generation call: it is handed the lineage of what has already been tried, the
knowledge the harness names, and the scorer itself, and it proposes only after
reading them.

    from awevolve import Harness, Lineage, AgentProposer, evolve

    harness = Harness(
        name="kernel",
        mutable_file="src/kernel.py",
        eval_command="python -m bench --strict",
        metric_regex=r"METRIC latency_ms=([0-9.]+)",
        metric_name="latency_ms",
        minimize=True,
    )
    result = evolve(harness, AgentProposer(), Lineage("lineage.jsonl"), max_rounds=20)

What you check afterwards is the lineage: every version, the score it earned, and
the edit that produced it — including the ones that lost, which are the rows that
stop the next proposal repeating a known failure.
"""

from .harness import (
    Harness,
    HarnessError,
    SelfScoringError,
    check_scorer_boundary,
    load_harness,
)
from .lineage import KEPT, PENDING, REFUSED, REVERTED, Lineage, Trial
from .loop import EvolveResult, Proposer, evolve
from .proposer import (
    NO_CHANGE,
    AgentError,
    AgentProposer,
    ScriptedProposer,
    build_prompt,
)
from .scorer import ScoreError, Scorer, ScoreResult
from .supervisor import Intervention, SupervisedProposer, Supervisor

__version__ = "0.1.0"

__all__ = [
    "AgentError",
    "AgentProposer",
    "EvolveResult",
    "Harness",
    "HarnessError",
    "Intervention",
    "KEPT",
    "Lineage",
    "NO_CHANGE",
    "PENDING",
    "Proposer",
    "REFUSED",
    "REVERTED",
    "ScoreError",
    "ScoreResult",
    "ScriptedProposer",
    "Scorer",
    "SelfScoringError",
    "SupervisedProposer",
    "Supervisor",
    "Trial",
    "__version__",
    "build_prompt",
    "check_scorer_boundary",
    "evolve",
    "load_harness",
]

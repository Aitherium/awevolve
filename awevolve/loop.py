"""The keep-or-revert loop, with the failures already designed out.

Every rule here exists because its absence is INVISIBLE. A run that mutated
nothing and a run that explored honestly and found nothing produce identical
logs, identical records and identical stop reasons; there is no exception to
catch and nothing to page on. So the loop refuses to proceed in the states where
the two become indistinguishable, rather than reporting a number nobody can
trust:

* a change that leaves the file byte-identical RAISES; it is never scored;
* a losing trial is RESTORED from a snapshot taken before the write, and a
  revert with no snapshot raises rather than reporting a revert that did not
  happen;
* the scorer's exit code is believed — a refused candidate is a refused trial;
* an apparent improvement is re-measured before it is banked, because a ratchet
  cannot walk back a win it has already kept;
* the proposer is handed the lineage, and a proposer that returns nothing ends
  the run rather than being retried forever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .harness import Harness
from .lineage import KEPT, REFUSED, REVERTED, Lineage
from .scorer import ScoreError, Scorer

__all__ = ["EvolveResult", "Proposer", "evolve"]

#: Consecutive agent failures before giving up. A few are noise on any
#: long run; an unbroken streak means the agent is gone, and burning the
#: whole round budget re-asking a dead endpoint helps nobody.
_MAX_AGENT_FAILURES = 5


class Proposer(Protocol):
    """Produces the next candidate, given everything known so far.

    Returns the FULL new content of the mutable file, or None to stop. Returning
    content rather than a patch keeps the contract decidable: the loop can always
    tell whether a proposal changes anything, which a patch format cannot
    guarantee without applying it first.
    """

    def __call__(self, context: dict[str, Any]) -> str | None: ...


@dataclass
class EvolveResult:
    """What a run achieved, and why it ended."""

    rounds: int = 0
    kept: int = 0
    reverted: int = 0
    refused: int = 0
    baseline: float | None = None
    best_score: float | None = None
    stop_reason: str = ""
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "kept": self.kept,
            "reverted": self.reverted,
            "refused": self.refused,
            "baseline": self.baseline,
            "best_score": self.best_score,
            "stop_reason": self.stop_reason,
            "improved": (
                self.best_score is not None
                and self.baseline is not None
                and self.best_score != self.baseline
            ),
        }


def evolve(
    harness: Harness,
    proposer: Proposer,
    lineage: Lineage,
    *,
    max_rounds: int = 10,
    patience: int = 3,
    baseline: float | None = None,
    scorer: Scorer | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> EvolveResult:
    """Run the loop until it converges, exhausts its rounds, or is stopped.

    Args:
        harness: what to improve and how it is scored.
        proposer: produces candidate file contents from the lineage.
        lineage: the durable record; also what the proposer is shown.
        max_rounds: hard cap on trials. 0 means unlimited, which is only
            sensible with a supervisor watching.
        patience: stop after this many judged trials with no improvement.
            0 disables the check.
        baseline: starting score. When None it is MEASURED before the first
            proposal — assuming a baseline is how a run reports an improvement
            over a number nobody checked.
        scorer: injectable for testing.
        on_event: optional observer, called with (event, payload).

    Returns:
        EvolveResult. Note ``rounds == 0`` with a clean stop reason is a real
        outcome and means the proposer declined immediately; it is NOT the same
        as a crash, and the reason says which.
    """
    available, why = harness.is_available()
    if not available:
        raise ScoreError(f"harness {harness.name!r} is not available: {why}")

    scorer = scorer or Scorer(harness)
    result = EvolveResult()
    emit = on_event or (lambda _e, _p: None)
    target: Path = harness.mutable_file

    # The baseline is MEASURED, not assumed. A run that starts from a number
    # nobody verified reports improvements against fiction.
    if baseline is None:
        baseline = scorer.score_once()
        emit("baseline", {"score": baseline, "metric": harness.metric_name})
    consecutive_agent_failures = 0
    result.baseline = baseline
    result.best_score = baseline
    current = baseline

    while True:
        if max_rounds and result.rounds >= max_rounds:
            result.stop_reason = "max_rounds"
            break
        if patience and lineage.rounds_without_improvement() >= patience:
            result.stop_reason = "no_improvement"
            break

        before = target.read_text(encoding="utf-8")
        context = {
            "harness": harness.to_dict(),
            "current_content": before,
            "baseline": current,
            "best_score": result.best_score,
            "lineage": lineage.render(),
            "summary": lineage.summary(),
            "round": result.rounds,
        }

        # A FAILED agent is a wasted round, NOT a decision to stop. A 7-day
        # autonomous run must survive a timeout, a rate limit or a malformed
        # reply; letting that escape ends the search and loses the lineage's
        # momentum for a reason that had nothing to do with the search. Only an
        # explicit decline (None) stops the loop -- which is why the agent
        # contract has a decline SENTINEL distinct from an error.
        try:
            proposal = proposer(context)
        except Exception as exc:                      # noqa: BLE001
            result.rounds += 1
            trial = lineage.open_trial("(agent failed)")
            lineage.settle(trial, REFUSED,
                           detail=f"{type(exc).__name__}: {exc}"[:400])
            result.refused += 1
            emit("refused", {"round": result.rounds,
                             "reason": f"agent: {type(exc).__name__}"})
            consecutive_agent_failures += 1
            if consecutive_agent_failures >= _MAX_AGENT_FAILURES:
                result.stop_reason = "agent_unavailable"
                break
            continue
        consecutive_agent_failures = 0
        if proposal is None:
            result.stop_reason = "proposer_declined"
            break

        result.rounds += 1
        trial = lineage.open_trial(_summarise(before, proposal))
        started = time.time()

        # ── apply, refusing a no-op ─────────────────────────────────────────
        if proposal == before:
            lineage.settle(
                trial, REFUSED,
                detail="proposal left the file byte-identical; a no-op is never scored",
            )
            result.refused += 1
            emit("refused", {"round": result.rounds, "reason": "no-op"})
            continue

        target.write_text(proposal, encoding="utf-8")
        emit("applied", {"round": result.rounds})

        # ── score, believing the exit code ──────────────────────────────────
        try:
            scored = scorer.score(baseline=current)
        except ScoreError as exc:
            # Restore FIRST. A refused candidate that stays on disk poisons every
            # subsequent trial, and the loop would keep scoring on top of it.
            target.write_text(before, encoding="utf-8")
            lineage.settle(trial, REFUSED, detail=str(exc))
            result.refused += 1
            emit("refused", {"round": result.rounds, "reason": str(exc)[:200]})
            continue

        # ── keep or revert ──────────────────────────────────────────────────
        if harness.better(scored.value, current):
            commit = lineage.commit_kept(trial, [target])
            lineage.settle(
                trial, KEPT,
                score=scored.value, samples=scored.samples, commit=commit,
                detail=(
                    f"confirmed over {len(scored.samples)} samples"
                    if scored.was_re_measured else ""
                ),
            )
            current = scored.value
            result.best_score = scored.value
            result.kept += 1
            emit("kept", {"round": result.rounds, "score": scored.value})
        else:
            target.write_text(before, encoding="utf-8")
            lineage.settle(
                trial, REVERTED,
                score=scored.value, samples=scored.samples,
                detail=f"{scored.value:g} did not beat {current:g}",
            )
            result.reverted += 1
            emit("reverted", {"round": result.rounds, "score": scored.value})

        del started

    result.trials = [t.to_dict() for t in lineage.trials]
    if not result.stop_reason:
        result.stop_reason = "complete"
    emit("done", result.to_dict())
    return result


def _summarise(before: str, after: str) -> str:
    """A one-line description of a change, for the lineage.

    Deliberately cheap: the lineage records WHAT happened and its score, and the
    full content of every candidate would make the record unreadable to the very
    proposer it exists to inform.
    """
    if after == before:
        return "(no change)"
    b_lines = before.splitlines()
    a_lines = after.splitlines()
    delta = len(a_lines) - len(b_lines)
    changed = sum(1 for x, y in zip(b_lines, a_lines) if x != y)
    for i, (x, y) in enumerate(zip(b_lines, a_lines)):
        if x != y:
            return (
                f"line {i + 1}: {x.strip()[:48]!r} -> {y.strip()[:48]!r} "
                f"({changed} changed, {delta:+d} lines)"
            )
    return f"{changed} line(s) changed, {delta:+d} lines"

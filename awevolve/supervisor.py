"""Noticing that a search has stopped searching, and doing something about it.

A long autonomous run fails in two ways that look identical from outside: the
agent exhausts its current line of thinking and proposes small variations of
something already rejected, or it cycles — proposing, being refused, and
proposing again. Both produce a steady stream of trials, a healthy-looking log,
and no improvement. Neither raises.

The supervisor watches the lineage for those shapes and intervenes by adding a
DIRECTIVE to the next prompt, rather than by stopping the run. Stopping is
already handled by ``patience``; the useful thing a supervisor adds is a fresh
angle before that limit is reached.

WHAT IT DOES NOT DO. It does not silently change the objective, relax the scorer,
or widen what the agent may edit. A supervisor that can move the goalposts turns
"the search stalled" into "the search found a way to look successful", which is
the failure this whole package is arranged to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .lineage import KEPT, REFUSED, REVERTED, Lineage

__all__ = ["Intervention", "Supervisor", "SupervisedProposer"]


@dataclass
class Intervention:
    """A steer, and the observation that justified it."""

    reason: str
    directive: str

    def render(self) -> str:
        return (
            "\n## Supervisor directive\n"
            f"The run has stalled: {self.reason}\n{self.directive}\n"
        )


class Supervisor:
    """Detects stagnation and produces a directive."""

    def __init__(
        self,
        *,
        stall_after: int = 3,
        refusal_streak: int = 3,
    ) -> None:
        """Args:
            stall_after: judged trials with no keep before steering.
            refusal_streak: consecutive refusals before steering. Refusals are
                treated separately from reverts because they mean something
                different: a revert is a real idea that lost, a refusal is a
                proposal that never became a measurement. A run producing only
                refusals is not exploring at all.
        """
        self.stall_after = stall_after
        self.refusal_streak = refusal_streak

    def assess(self, lineage: Lineage) -> Intervention | None:
        trials = lineage.trials
        if not trials:
            return None

        recent = [t for t in trials if t.outcome in (KEPT, REVERTED, REFUSED)]
        if not recent:
            return None

        # A streak of proposals that never became measurements.
        streak = 0
        for trial in reversed(recent):
            if trial.outcome != REFUSED:
                break
            streak += 1
        if streak >= self.refusal_streak:
            return Intervention(
                reason=(
                    f"the last {streak} proposals were refused without ever being "
                    f"scored"
                ),
                directive=(
                    "Stop refining the same edit. A refusal means the change was "
                    "empty or the scorer rejected it as incorrect - read the "
                    "refusal details above and fix the CAUSE. Propose a change of "
                    "a different KIND from anything in the history."
                ),
            )

        without = lineage.rounds_without_improvement()
        if without >= self.stall_after:
            tried = ", ".join(
                t.change_summary[:40] for t in recent[-self.stall_after:]
            )
            return Intervention(
                reason=f"{without} scored trials with no improvement",
                directive=(
                    "The recent attempts were variations on one idea: "
                    f"{tried}. Abandon that direction. Consider a different level "
                    "of the problem - the algorithm rather than its constants, the "
                    "data layout rather than the loop, the work avoided rather "
                    "than the work made faster."
                ),
            )
        return None


class SupervisedProposer:
    """Wraps a proposer, appending a directive when the run has stalled.

    A wrapper rather than a change to the loop: the loop's job is keep-or-revert,
    and steering is a property of the operator. Keeping them apart means a
    different proposer can be supervised without touching the loop, and the loop
    stays small enough to reason about.
    """

    def __init__(
        self,
        inner: Callable[[dict[str, Any]], str | None],
        lineage: Lineage,
        supervisor: Supervisor | None = None,
        *,
        on_intervene: Callable[[Intervention], None] | None = None,
    ) -> None:
        self.inner = inner
        self.lineage = lineage
        self.supervisor = supervisor or Supervisor()
        self._on_intervene = on_intervene
        self.interventions: list[Intervention] = []

    def __call__(self, context: dict[str, Any]) -> str | None:
        intervention = self.supervisor.assess(self.lineage)
        if intervention is not None:
            self.interventions.append(intervention)
            if self._on_intervene:
                self._on_intervene(intervention)
            context = dict(context)
            context["lineage"] = context.get("lineage", "") + intervention.render()
        return self.inner(context)

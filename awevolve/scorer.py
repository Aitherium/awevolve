"""Running the scorer, and believing the answer only when it earns it.

Two rules live here, and each one exists because its absence is invisible.

EXIT CODE. A scorer's non-zero exit is it saying the candidate is WRONG, not
merely slower. Parsing the metric out of stdout regardless makes that verdict
inert — and a broken build can still print an old-looking number, so the loop
scores the corpse and may keep it. An optimiser handed a pure speed reward finds
the wrong answer faster than a human would, and it looks like a win in every log.

CONFIRMATION. A ratchet cannot walk back a win it has already banked, so a single
noisy sample that happens to beat the baseline is kept permanently. Re-measuring
every trial would multiply the cost of an expensive harness by the repeat count,
so only a trial that CLAIMS a win is re-measured; losses need no confirmation and
wins are rare. The MEDIAN of the samples decides, so one outlier in either
direction cannot.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

from .harness import Harness

__all__ = ["ScoreError", "Scorer", "ScoreResult"]


class ScoreError(RuntimeError):
    """The scorer refused the candidate, or could not produce a number."""


@dataclass
class ScoreResult:
    """One believed score and how it was arrived at."""

    value: float
    samples: list[float]
    confirmed: bool

    @property
    def was_re_measured(self) -> bool:
        return len(self.samples) > 1


class Scorer:
    """Runs a harness's eval_command and returns a score worth acting on."""

    def __init__(
        self,
        harness: Harness,
        runner: Callable[[str, str, int], tuple[int, str]] | None = None,
    ) -> None:
        """``runner`` is injectable so the confirmation logic can be tested
        without spawning a process — the alternative is a test that asserts the
        arithmetic of a function it never actually runs."""
        self.harness = harness
        self._runner = runner or _run_command

    # ── one measurement ─────────────────────────────────────────────────────

    def score_once(self) -> float:
        """Score the CURRENT on-disk file exactly once."""
        h = self.harness
        code, output = self._runner(
            h.eval_command, str(h.get_base_dir()), h.time_budget_s
        )
        if code != 0:
            raise ScoreError(
                f"scorer exited {code}: the candidate is refused, not merely "
                f"slower. Last output: {output.strip()[-400:]}"
            )
        metric = h.extract_metric(output)
        if metric is None:
            raise ScoreError(
                f"scorer produced no {h.metric_name!r} metric matching "
                f"{h.metric_regex!r}. Output: {output.strip()[-400:]}"
            )
        return metric

    # ── a score worth banking ───────────────────────────────────────────────

    def score(self, baseline: float | None = None) -> ScoreResult:
        """Score, re-measuring only when the result CLAIMS an improvement.

        With no baseline there is nothing to claim against, so a single sample
        is returned — the first measurement of a run establishes the baseline
        rather than beating it.
        """
        h = self.harness
        first = self.score_once()
        if baseline is None or h.confirm_repeats <= 1:
            return ScoreResult(value=first, samples=[first], confirmed=False)
        if not h.better(first, baseline):
            # A loss needs no confirmation. Re-measuring it would multiply the
            # cost of every unsuccessful trial, which is most of them.
            return ScoreResult(value=first, samples=[first], confirmed=False)

        samples = [first]
        for _ in range(h.confirm_repeats - 1):
            samples.append(self.score_once())
        ordered = sorted(samples)
        median = ordered[len(ordered) // 2]
        return ScoreResult(value=median, samples=samples, confirmed=True)


def _run_command(command: str, cwd: str, timeout_s: int) -> tuple[int, str]:
    """Run a scorer and return (exit code, combined output).

    ``encoding`` is explicit: an undecodable byte from a crashing benchmark must
    not raise inside subprocess itself, because that failure would be reported as
    a scoring error rather than as the crash it is.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise ScoreError(f"scorer exceeded its {timeout_s}s budget") from None
    except OSError as exc:
        raise ScoreError(f"could not run the scorer: {exc}") from exc
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

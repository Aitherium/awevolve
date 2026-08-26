"""What an optimisable target is, and the one boundary that must never move.

A harness is a declarative description of something worth improving: ONE file an
optimiser may change, ONE command that scores it, and how to read the score back
out. That is the whole contract, and keeping it declarative is what lets a new
target be a data entry rather than an integration.

THE BOUNDARY. ``check_scorer_boundary`` refuses a harness whose mutable file is
also its scorer. If the file the optimiser may CHANGE is also the file that
REPORTS the score, the cheapest strategy available is not to improve the system,
it is to improve the report — and every downstream signal will agree
enthusiastically that things got better. It is not that an agent cheats out of
malice; it is that you defined a search space whose shortest path to a better
number runs through the measuring instrument, then asked something good at
finding shortest paths to search it.

This is checked at LOAD time, not at review time, because a self-scoring harness
is not a style problem — it invalidates every number the run produces.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Harness",
    "HarnessError",
    "SelfScoringError",
    "check_scorer_boundary",
    "load_harness",
]

#: One evaluation may not exceed this unless the harness says otherwise.
DEFAULT_TIME_BUDGET_S = 300


class HarnessError(ValueError):
    """A harness cannot be used as declared."""


class SelfScoringError(HarnessError):
    """The optimiser could rewrite what judges it."""


@dataclass
class Harness:
    """A scoreable target.

    Attributes:
        name: Identifier used in the lineage and on the CLI.
        mutable_file: The ONE file a trial may change.
        eval_command: Shell command that scores it. MUST exit non-zero when the
            candidate is incorrect, not merely slower — see ``strict``.
        metric_regex: Regex with exactly ONE capture group, anchored on the full
            metric name. A loose pattern that starts matching a neighbouring
            number is the worst failure this file can have: the loop keeps
            running, keeps reporting improvements, and is optimising something
            nobody chose.
        metric_name: Human-readable name, used in reports.
        minimize: True when lower is better.
        time_budget_s: Seconds one evaluation may take.
        confirm_repeats: Re-measurements required before believing an
            improvement. 1 trusts a single sample. Raise it wherever the score
            moves with machine load — a ratchet cannot walk back a win it has
            already banked.
        base_dir: Working directory for eval_command. Defaults to the mutable
            file's parent.
        knowledge: Files an agent should read before proposing. This is the
            paper's K, per harness rather than global.
    """

    name: str
    mutable_file: Path
    eval_command: str
    metric_regex: str
    metric_name: str = "metric"
    minimize: bool = False
    time_budget_s: int = DEFAULT_TIME_BUDGET_S
    confirm_repeats: int = 1
    base_dir: Path | None = None
    knowledge: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mutable_file = Path(self.mutable_file)
        if self.base_dir is not None:
            self.base_dir = Path(self.base_dir)
        self.knowledge = [Path(k) for k in self.knowledge]
        if self.confirm_repeats < 1:
            raise HarnessError("confirm_repeats must be at least 1")
        try:
            compiled = re.compile(self.metric_regex)
        except re.error as exc:
            raise HarnessError(f"metric_regex does not compile: {exc}") from exc
        if compiled.groups != 1:
            raise HarnessError(
                f"metric_regex must have exactly ONE capture group, has "
                f"{compiled.groups}: a pattern that captures the wrong number "
                f"optimises something nobody chose"
            )
        check_scorer_boundary(self)

    # ── availability ────────────────────────────────────────────────────────

    def is_available(self) -> tuple[bool, str]:
        """Whether this harness can run here, and why not when it cannot.

        Returns a reason rather than a bare False: an unavailable harness and a
        broken one look identical from the outside, and the difference decides
        whether a human should act.
        """
        if not self.mutable_file.exists():
            return False, f"mutable file not found: {self.mutable_file}"
        base = self.get_base_dir()
        if not base.exists():
            return False, f"base dir not found: {base}"
        return True, ""

    def get_base_dir(self) -> Path:
        if self.base_dir is not None:
            return self.base_dir
        return self.mutable_file.parent

    def extract_metric(self, output: str) -> float | None:
        """Read the score out of a scorer's output, or None if absent."""
        match = re.search(self.metric_regex, output)
        if not match:
            return None
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

    def better(self, score: float, baseline: float) -> bool:
        """Direction-aware comparison. Stated, never assumed.

        A minimised harness compared with a default ``>`` ratchets backwards
        while looking perfectly healthy: it keeps every trial that got worse.
        """
        return score < baseline if self.minimize else score > baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mutable_file": str(self.mutable_file),
            "eval_command": self.eval_command,
            "metric_regex": self.metric_regex,
            "metric_name": self.metric_name,
            "minimize": self.minimize,
            "time_budget_s": self.time_budget_s,
            "confirm_repeats": self.confirm_repeats,
            "base_dir": str(self.base_dir) if self.base_dir else None,
            "knowledge": [str(k) for k in self.knowledge],
        }


# ── the boundary ────────────────────────────────────────────────────────────

def check_scorer_boundary(harness: Harness) -> None:
    """Refuse a harness that could rewrite what judges it.

    Two shapes are refused, and both have been seen in real registries:

    * the mutable file is NAMED in its own eval_command — the training script
      that both trains and prints its own validation metric;
    * the mutable file IS the scorer script the command runs.

    Raises SelfScoringError. There is deliberately no override flag: a caller
    who genuinely wants this can split the scorer out of the mutable file, which
    takes minutes and is the actual fix. An escape hatch here would be used once
    in a hurry and then forever.
    """
    target = harness.mutable_file
    name = target.name
    if not name:
        return
    command = harness.eval_command

    # Tokens the command actually runs, so a substring coincidence in an
    # unrelated flag does not fire. shlex keeps quoted paths intact.
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        tokens = command.split()

    for token in tokens:
        stripped = token.strip("'\"")
        if not stripped:
            continue
        candidate = Path(stripped)
        if candidate.name != name:
            continue
        raise SelfScoringError(
            f"harness {harness.name!r} names its mutable file {name!r} in its "
            f"own eval_command ({command!r}). The optimiser could improve the "
            f"REPORT instead of the system, and every signal would agree it "
            f"worked. Split the scorer out of the mutable file."
        )


# ── loading ─────────────────────────────────────────────────────────────────

def load_harness(source: Path | str | dict[str, Any]) -> Harness:
    """Build a Harness from a JSON file or an already-parsed dict.

    Paths inside the document are resolved RELATIVE TO THE DOCUMENT, not to the
    current working directory, so a harness file means the same thing wherever
    it is invoked from.
    """
    if isinstance(source, dict):
        raw = dict(source)
        anchor = Path.cwd()
    else:
        path = Path(source)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise HarnessError(f"cannot read harness file {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HarnessError(f"harness file {path} is not valid JSON: {exc}") from exc
        anchor = path.parent

    missing = [k for k in ("name", "mutable_file", "eval_command", "metric_regex")
               if not raw.get(k)]
    if missing:
        raise HarnessError(f"harness is missing required field(s): {', '.join(missing)}")

    def _resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (anchor / candidate)

    return Harness(
        name=str(raw["name"]),
        mutable_file=_resolve(str(raw["mutable_file"])),
        eval_command=str(raw["eval_command"]),
        metric_regex=str(raw["metric_regex"]),
        metric_name=str(raw.get("metric_name", "metric")),
        minimize=bool(raw.get("minimize", False)),
        time_budget_s=int(raw.get("time_budget_s", DEFAULT_TIME_BUDGET_S)),
        confirm_repeats=int(raw.get("confirm_repeats", 1)),
        base_dir=_resolve(str(raw["base_dir"])) if raw.get("base_dir") else None,
        knowledge=[_resolve(str(k)) for k in raw.get("knowledge", [])],
    )

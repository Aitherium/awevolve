"""The record of what was tried, what it scored, and what came of it.

This is the paper's P_t, and it is the input a variation operator actually needs.
A proposer that cannot see what has already been tried is sampling, not
searching: it will re-propose a change that was measured and reverted an hour
ago, with no way to know.

Append-only JSONL, one line per trial, written with an atomic replace of the
whole file so a crashed run cannot leave a half-written record that the next run
reads as truth. Small by construction — a trial is a few hundred bytes, and a
seven-day run is thousands of lines, not millions.

WHY OUTCOMES ARE WRITTEN BACK. A trial's score is known before its fate is: the
loop scores, then decides. Recording only proposals would leave a lineage that
cannot answer the one question a proposer needs answered — "did that work?" —
so ``settle`` re-writes the trial with its outcome.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("awevolve.lineage")

__all__ = ["Trial", "Lineage"]

#: A trial that has been proposed and applied but not yet judged.
PENDING = "pending"
KEPT = "kept"
REVERTED = "reverted"
REFUSED = "refused"   # never scored: a no-op, or the scorer rejected it

_TERMINAL = {KEPT, REVERTED, REFUSED}


@dataclass
class Trial:
    """One attempt, and what became of it."""

    index: int
    change_summary: str
    #: Unique per trial, so concurrent writers never share a filename. `index`
    #: is a display ordinal and is NOT unique when several proposers open a
    #: trial at the same moment.
    trial_id: str = ""
    outcome: str = PENDING
    score: float | None = None
    samples: list[float] = field(default_factory=list)
    detail: str = ""
    commit: str = ""
    started_at: float = 0.0
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Trial":
        known = {f for f in cls.__dataclass_fields__}  # noqa: F821
        return cls(**{k: v for k, v in raw.items() if k in known})


class Lineage:
    """The committed sequence of trials for one evolution session."""

    def __init__(self, path: Path | str, *, git_root: Path | str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ONE FILE PER TRIAL, in a directory beside the lineage. Two writers
        # touch two different paths, so neither can clobber the other -- the
        # same reason awrun/store.py keeps one file per queue item. A whole-file
        # rewrite lost 7 of 8 concurrent trials when measured.
        self.dir = self.path.with_suffix(self.path.suffix + ".d")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.git_root = Path(git_root) if git_root else None
        self._own: dict[str, Trial] = {}

    # ── reading ─────────────────────────────────────────────────────────────

    def _read(self) -> Iterator[Trial]:
        """Every trial on disk, from every writer -- not just this one's.

        Read fresh on each access rather than cached, because with several
        proposers working one lineage a cached list is stale the moment another
        one settles a trial, and a proposer shown a stale history re-proposes
        what somebody already measured.
        """
        for f in sorted(self.dir.glob("*.json")):
            try:
                yield Trial.from_dict(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                # A corrupt or half-written record is skipped, never fatal:
                # losing one trial must not make a whole run unreadable.
                continue

    @property
    def trials(self) -> list[Trial]:
        # Ordered by when the trial STARTED, with the id as a tiebreak so the
        # order is total and stable. Not by a shared counter: two proposers
        # opening a trial at the same instant would be handed the same index.
        return sorted(self._read(), key=lambda t: (t.started_at, t.trial_id))

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.glob("*.json"))

    @property
    def kept(self) -> list[Trial]:
        return [t for t in self.trials if t.outcome == KEPT]

    @property
    def best(self) -> Trial | None:
        scored = [t for t in self.kept if t.score is not None]
        return scored[-1] if scored else None

    def rounds_without_improvement(self) -> int:
        """How many judged trials since the last kept one.

        This is the stagnation signal the supervisor acts on.
        """
        count = 0
        for trial in reversed(self.trials):
            if trial.outcome == KEPT:
                return count
            if trial.outcome in _TERMINAL:
                count += 1
        return count

    def summary(self) -> dict[str, Any]:
        return {
            "trials": len(self),
            "kept": len(self.kept),
            "reverted": sum(1 for t in self.trials if t.outcome == REVERTED),
            "refused": sum(1 for t in self.trials if t.outcome == REFUSED),
            "best_score": self.best.score if self.best else None,
            "rounds_without_improvement": self.rounds_without_improvement(),
        }

    # ── writing ─────────────────────────────────────────────────────────────

    def _write(self, trial: Trial) -> None:
        """Persist ONE trial to ITS OWN file. Never rewrites another writer's.

        The temp file carries the trial id, so two writers flushing at the same
        moment do not contend for one temp path -- on Windows `os.replace`
        raises PermissionError when another handle holds the target, so a shared
        temp name turns a race into a crash rather than a lost write.
        """
        target = self.dir / f"{trial.trial_id}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(trial.to_dict()), encoding="utf-8")
        # Retry: on Windows os.replace fails with PermissionError when ANY
        # other handle holds the target, and with several proposers reading the
        # lineage that happens routinely. A few short retries turn a lost trial
        # into a slightly slower one.
        for attempt in range(6):
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.02 * (attempt + 1))
        # NOT _render() here. Rendering globs and READS every trial file, so
        # doing it on every write gave each writer N concurrent readers to
        # contend with -- 5 of 16 writers then failed their own os.replace.
        # The per-trial files are the source of truth; the .jsonl view is
        # rendered when somebody actually asks for it.

    def _render(self) -> None:
        """Rewrite the .jsonl as a READ-ONLY view of the directory.

        Best-effort and never fatal: it is a convenience for readers and for
        `awevolve lineage`, and the per-trial files are the source of truth. A
        writer that loses the race to render has still recorded its trial.
        """
        try:
            body = "\n".join(json.dumps(t.to_dict()) for t in self.trials)
            # uuid, not pid: several proposers in ONE process share a pid, so a
            # pid-named temp is the same shared path and the race comes
            # straight back. Measured: 2 of 8 threads still crashed.
            tmp = self.path.with_suffix(
                self.path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text(body + ("\n" if body else ""), encoding="utf-8")
            os.replace(tmp, self.path)
        except (OSError, ValueError, TypeError) as exc:
            # The per-trial files are the source of truth, so a failed render
            # loses a convenience and never a trial -- but it is SAID, not
            # swallowed. A handler whose whole body is `pass` is how "the view
            # is stale" becomes indistinguishable from "there were no trials".
            logger.debug("lineage view not rendered (%s: %s)",
                         type(exc).__name__, exc)

    def open_trial(self, change_summary: str) -> Trial:
        trial = Trial(
            index=len(self),
            change_summary=change_summary,
            started_at=time.time(),
            trial_id=uuid.uuid4().hex[:12],
        )
        self._own[trial.trial_id] = trial
        self._write(trial)
        return trial

    def settle(
        self,
        trial: Trial,
        outcome: str,
        *,
        score: float | None = None,
        samples: list[float] | None = None,
        detail: str = "",
        commit: str = "",
    ) -> Trial:
        """Record what became of a trial.

        A trial whose fate is never written stays ``pending`` forever and is
        indistinguishable from a run that is still going — which is exactly how
        a crashed loop reads as a healthy one.
        """
        if outcome not in _TERMINAL:
            raise ValueError(f"outcome must be one of {sorted(_TERMINAL)}, got {outcome!r}")
        trial.outcome = outcome
        trial.score = score
        trial.samples = list(samples or ([] if score is None else [score]))
        trial.detail = detail
        trial.commit = commit
        trial.duration_s = round(time.time() - trial.started_at, 3)
        self._write(trial)
        return trial

    # ── the durable half ────────────────────────────────────────────────────

    def commit_kept(self, trial: Trial, paths: list[Path]) -> str:
        """Commit a kept version, when a git root was supplied.

        Returns the short sha, or "" when git is unavailable or there was
        nothing to commit. Deliberately non-fatal: a lineage that records the
        trial is still useful without a commit, and a run should not die because
        a repository is in an unexpected state. The JSONL is the source of
        truth; git is the durable convenience.

        Only the paths given are staged. Committing everything in a working tree
        would sweep in whatever else happens to be uncommitted — which, in a tree
        other people are also writing to, is somebody else's unfinished work.
        """
        if self.git_root is None or not paths:
            return ""
        try:
            subprocess.run(
                ["git", "add", "--", *[str(p) for p in paths]],
                cwd=str(self.git_root), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60, check=True,
            )
            message = (
                f"awevolve: trial {trial.index} kept "
                f"({trial.change_summary[:60]})"
            )
            done = subprocess.run(
                ["git", "commit", "-m", message, "--", *[str(p) for p in paths]],
                cwd=str(self.git_root), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
            )
            if done.returncode != 0:
                return ""
            sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.git_root), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=30,
            )
            return sha.stdout.strip() if sha.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    # ── what a proposer is shown ────────────────────────────────────────────

    def render(self, limit: int = 20) -> str:
        """The lineage as text a proposer can read.

        Reverted and refused trials are included ON PURPOSE and are the most
        valuable rows here: "this was tried and it scored worse" is the single
        thing that stops an agent re-proposing it, and a lineage of successes
        only would invite exactly that.
        """
        if not len(self):
            return "(no trials yet — this is the first proposal)"
        lines = []
        for trial in self.trials[-limit:]:
            score = "n/a" if trial.score is None else f"{trial.score:g}"
            lines.append(
                f"#{trial.index} [{trial.outcome}] score={score} "
                f"{trial.change_summary[:100]}"
                + (f" — {trial.detail[:120]}" if trial.detail else "")
            )
        return "\n".join(lines)

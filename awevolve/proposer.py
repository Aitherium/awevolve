"""The variation operator: an agent, not a single generation call.

This is the whole point of the package. Classical evolutionary search confines a
model to one generation call inside a fixed pipeline — sample parents, ask for a
child, score it, repeat — and that confinement is the ceiling. The agent here is
handed the lineage, the knowledge, and the current file, and is expected to have
already reasoned about what failed before proposing anything.

NO AGENT IS BUNDLED, and that is deliberate. Bundling one would mean depending on
a particular SDK, which makes this package useless to anyone who runs a different
one; and the honest interface to "an agent" is a command that reads a prompt and
writes a file. So the default proposer SHELLS OUT to a command you configure,
and any callable matching the ``Proposer`` protocol can replace it.

WHAT THE AGENT IS SHOWN, and why each part is there:

* the current file — it is editing this, not describing an edit;
* the lineage INCLUDING reverted and refused trials — "this was tried and scored
  worse" is the single most valuable line in the prompt, and a history of
  successes only invites re-proposing a known failure;
* the knowledge files the harness names — the paper's K;
* the metric, its direction, and the current baseline — an optimiser that does
  not know which way is better is guessing.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

__all__ = ["AgentProposer", "ScriptedProposer", "build_prompt"]

#: The agent command. Reads the prompt on stdin, writes the full new file to
#: stdout. Anything that satisfies that contract works.
AGENT_CMD_ENV = "AWEVOLVE_AGENT_CMD"

DEFAULT_TIMEOUT_S = 900

_PROMPT = """\
You are the variation operator in an evolutionary search. Your job is to propose
ONE change to a single file that improves a measured score.

## The objective
metric        : {metric_name}
direction     : {direction}
current score : {baseline}
best so far   : {best}

## The file you may change
{path}

## What has already been tried
{lineage}

Read that carefully. A trial marked [reverted] scored WORSE and re-proposing it
wastes a round. A trial marked [refused] never scored at all - either it changed
nothing, or the scorer rejected it as incorrect.

{knowledge}
## The current contents
```
{content}
```

## Your answer
Output the COMPLETE new contents of the file and nothing else - no explanation,
no markdown fence, no diff. If you have no change worth making, output exactly:
{no_change}

A change that leaves the file identical is refused and costs a round, so do not
output the file unchanged as a way of saying "nothing to do".
"""

#: What a proposer says when it has nothing worth trying. A sentinel rather than
#: an empty response: empty output is indistinguishable from a crashed agent, and
#: the two must not lead to the same decision.
NO_CHANGE = "AWEVOLVE_NO_CHANGE"


def build_prompt(context: dict[str, Any]) -> str:
    """Render what the agent sees. Pure, so it can be asserted directly."""
    harness = context.get("harness", {})
    knowledge = _render_knowledge(harness.get("knowledge") or [])
    baseline = context.get("baseline")
    best = context.get("best_score")
    return _PROMPT.format(
        metric_name=harness.get("metric_name", "metric"),
        direction="lower is better" if harness.get("minimize") else "higher is better",
        baseline="unknown" if baseline is None else f"{baseline:g}",
        best="unknown" if best is None else f"{best:g}",
        path=harness.get("mutable_file", "(unknown)"),
        lineage=context.get("lineage", "(none)"),
        knowledge=knowledge,
        content=context.get("current_content", ""),
        no_change=NO_CHANGE,
    )


def _render_knowledge(paths: list[str]) -> str:
    """Inline the harness's knowledge files.

    Unreadable entries are NAMED rather than skipped: an agent told a reference
    exists and silently not given it will reason as though it had read it.
    """
    if not paths:
        return ""
    chunks = ["## Reference material"]
    for raw in paths:
        path = Path(raw)
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            chunks.append(f"### {path.name}\n(could not be read: {exc})")
            continue
        chunks.append(f"### {path.name}\n```\n{body}\n```")
    return "\n\n".join(chunks) + "\n"


class ScriptedProposer:
    """A fixed sequence of proposals. For tests and for replaying a lineage."""

    def __init__(self, proposals: list[str | None]) -> None:
        self._queue = list(proposals)
        self.seen: list[dict[str, Any]] = []

    def __call__(self, context: dict[str, Any]) -> str | None:
        self.seen.append(context)
        if not self._queue:
            return None
        return self._queue.pop(0)


class AgentProposer:
    """Runs a coding agent as the variation operator.

    The command receives the prompt on stdin and must write the complete new file
    to stdout. That contract is intentionally the lowest common denominator: any
    agent CLI, any script, any HTTP shim wrapped in three lines of shell.
    """

    def __init__(
        self,
        command: str | None = None,
        *,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        runner: Callable[[list[str], str, int], tuple[int, str, str]] | None = None,
    ) -> None:
        self.command = command or os.getenv(AGENT_CMD_ENV, "")
        if not self.command:
            raise ValueError(
                f"no agent command: pass command= or set {AGENT_CMD_ENV}. It must "
                f"read a prompt on stdin and write the complete new file to "
                f"stdout."
            )
        self.timeout_s = timeout_s
        self._runner = runner or _run_agent
        self.last_stderr = ""

    def __call__(self, context: dict[str, Any]) -> str | None:
        prompt = build_prompt(context)
        argv = shlex.split(self.command, posix=False)
        code, out, err = self._runner(argv, prompt, self.timeout_s)
        self.last_stderr = err

        if code != 0:
            # A failed agent is not a decision to stop. It is a round that
            # produced nothing, and saying so is different from the agent
            # deliberately declining -- which is why this returns a no-op marker
            # the loop refuses rather than None, which would end the run.
            raise AgentError(f"agent exited {code}: {err.strip()[-400:]}")

        answer = _strip_fence(out.strip())
        if not answer:
            raise AgentError(
                "agent produced no output. An empty response and a crashed agent "
                "are indistinguishable here, so this is an error rather than a "
                f"silent decline -- use {NO_CHANGE} to decline."
            )
        if answer.strip() == NO_CHANGE:
            return None
        return answer


class AgentError(RuntimeError):
    """The agent could not produce a candidate."""


def _strip_fence(text: str) -> str:
    """Remove a markdown fence an agent added despite being asked not to.

    Tolerated rather than rejected: a fence is the single most common deviation,
    it is unambiguous to strip, and failing the round over it would spend a real
    evaluation budget on a formatting nit.
    """
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


def _run_agent(argv: list[str], prompt: str, timeout_s: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise AgentError(f"agent exceeded its {timeout_s}s budget") from None
    except OSError as exc:
        raise AgentError(f"could not run the agent {argv[0]!r}: {exc}") from exc
    return proc.returncode, proc.stdout or "", proc.stderr or ""

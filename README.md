# awevolve — an agent as the variation operator

Point it at a file and a command that scores that file, and watch an agent
improve it — keeping every version and the score it earned.

```bash
pip install awevolve
```

Python 3.10+. **No dependencies.** The agent is a command you configure, so this
works with whatever coding agent you already run instead of pinning you to one
SDK.

## What it is

Classical evolutionary search confines a model to a single generation call inside
a fixed pipeline: sample some parents, ask for a child, score it, repeat. That
confinement is the ceiling. `awevolve` hands the agent the **lineage** of what
has already been tried, the knowledge the harness names, and the scorer itself —
and expects it to have read them before proposing anything.

```python
from awevolve import Harness, Lineage, AgentProposer, evolve

harness = Harness(
    name="kernel",
    mutable_file="src/kernel.py",
    eval_command="python -m bench --strict",
    metric_regex=r"METRIC latency_ms=([0-9.]+)",
    metric_name="latency_ms",
    minimize=True,
    confirm_repeats=3,          # this score moves with machine load
)

result = evolve(harness, AgentProposer(), Lineage("kernel.jsonl"), max_rounds=20)
```

```bash
awevolve check   --harness kernel.json          # validate; changes nothing
awevolve run     --harness kernel.json --rounds 20
awevolve lineage --lineage kernel.jsonl         # what was tried, and what it scored
awevolve status  --lineage kernel.jsonl
awevolve --self-test                            # prove the invariants still hold
```

## Instead of trusting

*that your optimisation loop is finding anything.*

## You check

*every version it kept, the score that version earned, and the edit that produced
it* — including the ones that lost, which are the rows that stop the next
proposal repeating a known failure.

## The agent contract

One command. It reads a prompt on stdin and writes the **complete new file** to
stdout:

```bash
export AWEVOLVE_AGENT_CMD="my-agent --print"
```

To decline a round it outputs exactly `AWEVOLVE_NO_CHANGE`. That is a sentinel
rather than empty output on purpose: an empty response and a crashed agent are
indistinguishable, and they must not lead to the same decision.

Any callable works too — `AgentProposer` is just the default:

```python
def my_proposer(context: dict) -> str | None:
    ...  # context carries current_content, lineage, baseline, harness
```

## The failures this is built around

Every rule below exists because its absence is **invisible**. A run that mutated
nothing and a run that explored honestly and found nothing produce identical
logs, identical records and identical stop reasons. There is no exception to
catch and nothing to alert on. So the loop refuses to proceed in the states where
the two become indistinguishable:

| rule | why |
|---|---|
| A change that leaves the file byte-identical is **refused**, never scored | otherwise a no-op is recorded as a trial that "found no improvement" |
| A losing trial is **restored from a snapshot** taken before the write | a revert that only logs keeps every regression it ever measured |
| A **non-zero exit** from the scorer refuses the trial | that is the scorer saying the candidate is *wrong*, not slower; a broken build still prints an old-looking number |
| An apparent win is **re-measured** before it is banked | a ratchet cannot walk back a win it has already kept, so one lucky sample is permanent |
| The baseline is **measured**, not assumed | a run that starts from an unverified number reports improvements against fiction |
| The proposer is handed the **lineage** | a proposer that cannot see what failed is sampling, not searching |

Losses are not re-measured — that would multiply the cost of every unsuccessful
trial, which is most of them. Only a trial that *claims* a win pays for
confirmation, and the **median** of the samples decides so one outlier in either
direction cannot.

## The one rule that matters most

**A harness may not be its own scorer.** This is checked at construction and
there is deliberately no override:

```python
Harness(
    mutable_file="train.py",
    eval_command="python train.py",   # SelfScoringError
    ...
)
```

If the file the optimiser may **change** is also the file that **reports** the
score, the cheapest available strategy is not to improve the system — it is to
improve the report. And every downstream signal will agree enthusiastically that
things got better.

It is not that an agent cheats out of malice. It is that you have defined a
search space whose shortest path to a better number runs straight through the
measuring instrument, and then asked something very good at finding shortest
paths to search it.

The fix takes minutes: split the scorer out of the mutable file. An escape hatch
here would be used once in a hurry and then forever.

## Supervision

Long autonomous runs stall in two ways that look identical from outside — the
agent exhausts a line of thinking and proposes variations on something already
rejected, or it cycles through proposals that are never scored at all. Both
produce a steady stream of trials and no improvement.

```python
from awevolve import SupervisedProposer, Supervisor

proposer = SupervisedProposer(AgentProposer(), lineage, Supervisor(stall_after=3))
```

The supervisor adds a **directive** to the next prompt. It never changes the
objective, relaxes the scorer, or widens what may be edited — a supervisor that
can move the goalposts turns "the search stalled" into "the search found a way to
look successful".

## The lineage

Append-only JSONL, one line per trial, written with an atomic replace so a
crashed run cannot leave a half-written record the next run reads as truth. Pass
`git_root=` and each kept version is committed too — staging **only** the mutable
file, because committing a whole working tree sweeps in whatever else happened to
be uncommitted.

## Licence

Apache-2.0.

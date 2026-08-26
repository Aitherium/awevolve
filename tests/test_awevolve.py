"""awevolve's invariants, asserted against the bytes on disk.

Every test here drives the REAL loop. A suite that reimplements the loop it is
testing passes forever no matter what the loop does — which is exactly how the
predecessor to this package ran for weeks having mutated nothing while reporting
success on every round.

The scorer is a deterministic function of the mutable file's contents, so
keep/revert decisions are reproducible rather than depending on a live benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from awevolve import (
    NO_CHANGE,
    AgentError,
    AgentProposer,
    Harness,
    HarnessError,
    Lineage,
    ScriptedProposer,
    SelfScoringError,
    SupervisedProposer,
    Supervisor,
    build_prompt,
    evolve,
)
from awevolve.scorer import ScoreError, Scorer

SCORER = (
    "import io, re\n"
    "s = io.open('config.py', encoding='utf-8').read()\n"
    "m = re.search(r'score\\s*=\\s*([0-9.]+)', s)\n"
    "print('METRIC v=' + (m.group(1) if m else '0'))\n"
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "config.py").write_text("score = 5\n", encoding="utf-8")
    (tmp_path / "score.py").write_text(SCORER, encoding="utf-8")
    return tmp_path


@pytest.fixture
def harness(workdir: Path) -> Harness:
    return Harness(
        name="t",
        mutable_file=workdir / "config.py",
        eval_command=f"{sys.executable} score.py",
        metric_regex=r"METRIC v=([0-9.]+)",
        metric_name="v",
        minimize=False,
        base_dir=workdir,
        time_budget_s=60,
    )


# =============================================================================
# The scorer boundary — the invariant a self-improving loop needs most
# =============================================================================


def test_a_harness_may_not_be_its_own_scorer(workdir: Path) -> None:
    """If the optimiser may change the file that reports the score, the cheapest
    way to win is to change the report — and every signal agrees it worked."""
    with pytest.raises(SelfScoringError, match="own eval_command"):
        Harness(
            name="self",
            mutable_file=workdir / "score.py",
            eval_command=f"{sys.executable} score.py",
            metric_regex=r"METRIC v=([0-9.]+)",
            base_dir=workdir,
        )


def test_the_boundary_does_not_cry_wolf(workdir: Path) -> None:
    """A harness whose scorer is a different file is fine, even when the names
    are similar — a rule that floods gets switched off."""
    (workdir / "config_bench.py").write_text("print('METRIC v=1')\n", encoding="utf-8")
    Harness(
        name="ok",
        mutable_file=workdir / "config.py",
        eval_command=f"{sys.executable} config_bench.py",
        metric_regex=r"METRIC v=([0-9.]+)",
        base_dir=workdir,
    )


def test_a_regex_without_exactly_one_group_is_refused(workdir: Path) -> None:
    """A pattern that captures the wrong number optimises something nobody
    chose, and the loop keeps reporting improvements while it does."""
    with pytest.raises(HarnessError, match="ONE capture group"):
        Harness(
            name="t", mutable_file=workdir / "config.py",
            eval_command=f"{sys.executable} score.py",
            metric_regex=r"METRIC v=[0-9.]+",
            base_dir=workdir,
        )


# =============================================================================
# The loop acts on disk
# =============================================================================


def test_a_winning_change_reaches_disk(harness: Harness, workdir: Path) -> None:
    lineage = Lineage(workdir / "l.jsonl")
    result = evolve(harness, ScriptedProposer(["score = 9\n"]), lineage,
                    max_rounds=1, patience=0)
    assert result.kept == 1
    assert "score = 9" in (workdir / "config.py").read_text(encoding="utf-8")
    assert result.best_score == 9.0


def test_a_losing_change_is_restored(harness: Harness, workdir: Path) -> None:
    """A ratchet that reports a revert without performing one keeps every
    regression it ever measured."""
    before = (workdir / "config.py").read_text(encoding="utf-8")
    lineage = Lineage(workdir / "l.jsonl")
    result = evolve(harness, ScriptedProposer(["score = 1\n"]), lineage,
                    max_rounds=1, patience=0)
    assert result.reverted == 1
    assert (workdir / "config.py").read_text(encoding="utf-8") == before


def test_a_no_op_is_refused_and_never_scored(harness: Harness, workdir: Path) -> None:
    lineage = Lineage(workdir / "l.jsonl")
    result = evolve(harness, ScriptedProposer(["score = 5\n"]), lineage,
                    max_rounds=1, patience=0)
    assert (result.refused, result.kept, result.reverted) == (1, 0, 0)
    assert lineage.trials[0].score is None, "a no-op was given a score"


def test_a_refused_candidate_does_not_stay_on_disk(workdir: Path) -> None:
    """A candidate the scorer rejects must be removed before the next trial, or
    every later measurement is taken on top of it."""
    harness = Harness(
        name="t", mutable_file=workdir / "config.py",
        eval_command=f"{sys.executable} score.py && exit 3",
        metric_regex=r"METRIC v=([0-9.]+)",
        base_dir=workdir, time_budget_s=60,
    )
    before = (workdir / "config.py").read_text(encoding="utf-8")
    lineage = Lineage(workdir / "l.jsonl")
    result = evolve(harness, ScriptedProposer(["score = 9\n"]), lineage,
                    max_rounds=1, patience=0, baseline=5.0)
    assert result.refused == 1
    assert (workdir / "config.py").read_text(encoding="utf-8") == before


def test_the_baseline_is_measured_not_assumed(harness: Harness, workdir: Path) -> None:
    """A run that starts from an unverified number reports improvements against
    fiction."""
    lineage = Lineage(workdir / "l.jsonl")
    result = evolve(harness, ScriptedProposer([None]), lineage, max_rounds=1)
    assert result.baseline == 5.0
    assert result.stop_reason == "proposer_declined"


# =============================================================================
# The lineage is the input a variation operator needs
# =============================================================================


def test_the_proposer_is_shown_what_failed(harness: Harness, workdir: Path) -> None:
    """Reverted trials are the most valuable rows in the history: they are what
    stops the next proposal repeating a known failure."""
    sp = ScriptedProposer(["score = 1\n", "score = 9\n"])
    lineage = Lineage(workdir / "l.jsonl")
    evolve(harness, sp, lineage, max_rounds=2, patience=0)
    second = sp.seen[1]["lineage"]
    assert "reverted" in second
    assert "score = 1" in second or "-> 'score = 1'" in second


def test_outcomes_are_written_back(harness: Harness, workdir: Path) -> None:
    """A trial left pending forever is indistinguishable from a run still in
    progress, which is how a crashed loop reads as a healthy one."""
    lineage = Lineage(workdir / "l.jsonl")
    evolve(harness, ScriptedProposer(["score = 9\n"]), lineage, max_rounds=1, patience=0)
    reread = Lineage(workdir / "l.jsonl")
    assert [t.outcome for t in reread.trials] == ["kept"]
    assert reread.summary()["kept"] == 1


def test_stagnation_is_counted_from_judged_trials(harness: Harness, workdir: Path) -> None:
    lineage = Lineage(workdir / "l.jsonl")
    evolve(harness, ScriptedProposer(["score = 1\n", "score = 2\n", "score = 3\n"]),
           lineage, max_rounds=3, patience=0)
    assert lineage.rounds_without_improvement() == 3


def test_patience_stops_a_stalled_run(harness: Harness, workdir: Path) -> None:
    lineage = Lineage(workdir / "l.jsonl")
    result = evolve(
        harness,
        ScriptedProposer(["score = 1\n", "score = 2\n", "score = 3\n", "score = 9\n"]),
        lineage, max_rounds=10, patience=2,
    )
    assert result.stop_reason == "no_improvement"
    assert result.kept == 0


# =============================================================================
# Confirmation — a lucky sample must not be banked
# =============================================================================


def test_an_apparent_win_is_re_measured(workdir: Path) -> None:
    """A ratchet cannot walk back a win it has kept, so one noisy sample that
    beats the baseline would be permanent."""
    calls = {"n": 0}
    seq = [99.0, 1.0, 1.0]

    def runner(_cmd: str, _cwd: str, _t: int) -> tuple[int, str]:
        i = calls["n"]
        calls["n"] += 1
        return 0, f"METRIC v={seq[i] if i < len(seq) else 1.0}"

    harness = Harness(
        name="noisy", mutable_file=workdir / "config.py",
        eval_command=f"{sys.executable} score.py",
        metric_regex=r"METRIC v=([0-9.]+)",
        base_dir=workdir, confirm_repeats=3, time_budget_s=60,
    )
    scorer = Scorer(harness, runner=runner)
    result = scorer.score(baseline=5.0)
    assert result.value == 1.0, "the 99.0 fluke was believed"
    assert result.was_re_measured


def test_a_loss_is_not_re_measured(workdir: Path) -> None:
    """Re-measuring losses multiplies the cost of every unsuccessful trial,
    which is most of them."""
    calls = {"n": 0}

    def runner(_cmd: str, _cwd: str, _t: int) -> tuple[int, str]:
        calls["n"] += 1
        return 0, "METRIC v=1.0"

    harness = Harness(
        name="noisy", mutable_file=workdir / "config.py",
        eval_command=f"{sys.executable} score.py",
        metric_regex=r"METRIC v=([0-9.]+)",
        base_dir=workdir, confirm_repeats=3, time_budget_s=60,
    )
    Scorer(harness, runner=runner).score(baseline=5.0)
    assert calls["n"] == 1


def test_a_nonzero_exit_is_a_refusal_not_a_score(workdir: Path, harness: Harness) -> None:
    def runner(_cmd: str, _cwd: str, _t: int) -> tuple[int, str]:
        return 1, "METRIC v=999"

    with pytest.raises(ScoreError, match="refused"):
        Scorer(harness, runner=runner).score_once()


# =============================================================================
# The agent contract
# =============================================================================


def test_the_prompt_carries_the_direction_and_the_history() -> None:
    prompt = build_prompt({
        "harness": {"metric_name": "lat", "minimize": True, "mutable_file": "k.py"},
        "baseline": 3.0, "best_score": 3.0,
        "lineage": "#0 [reverted] score=9 tried unrolling",
        "current_content": "x = 1",
    })
    assert "lower is better" in prompt
    assert "reverted" in prompt
    assert "x = 1" in prompt
    assert NO_CHANGE in prompt


def test_an_agent_declines_with_a_sentinel_not_silence() -> None:
    """Empty output and a crashed agent are indistinguishable, so they must not
    lead to the same decision."""
    def runner(_argv, _prompt, _t):
        return 0, NO_CHANGE, ""

    assert AgentProposer("x", runner=runner)({"harness": {}}) is None

    def empty(_argv, _prompt, _t):
        return 0, "", ""

    with pytest.raises(AgentError, match="no output"):
        AgentProposer("x", runner=empty)({"harness": {}})


def test_a_failed_agent_raises_rather_than_ending_the_run() -> None:
    """A crashed agent is a wasted round, not a decision to stop searching."""
    def boom(_argv, _prompt, _t):
        return 1, "", "boom"

    with pytest.raises(AgentError, match="exited 1"):
        AgentProposer("x", runner=boom)({"harness": {}})


def test_a_markdown_fence_is_tolerated() -> None:
    def fenced(_argv, _prompt, _t):
        return 0, "```\nscore = 9\n```", ""

    assert AgentProposer("x", runner=fenced)({"harness": {}}) == "score = 9"


def test_an_agent_with_no_command_is_refused() -> None:
    with pytest.raises(ValueError, match="AWEVOLVE_AGENT_CMD"):
        AgentProposer("")


# =============================================================================
# Supervision
# =============================================================================


def test_the_supervisor_steers_a_stalled_run(harness: Harness, workdir: Path) -> None:
    lineage = Lineage(workdir / "l.jsonl")
    inner = ScriptedProposer(["score = 1\n", "score = 2\n", "score = 3\n", "score = 4\n"])
    supervised = SupervisedProposer(inner, lineage, Supervisor(stall_after=2))
    evolve(harness, supervised, lineage, max_rounds=4, patience=0)
    assert supervised.interventions, "a stalled run was never steered"
    assert "no improvement" in supervised.interventions[0].reason
    assert "Supervisor directive" in inner.seen[-1]["lineage"]


def test_the_supervisor_is_quiet_on_a_healthy_run(harness: Harness, workdir: Path) -> None:
    """A supervisor that always intervenes is the same failure as a gate that
    always passes."""
    lineage = Lineage(workdir / "l.jsonl")
    inner = ScriptedProposer(["score = 6\n", "score = 7\n"])
    supervised = SupervisedProposer(inner, lineage, Supervisor(stall_after=2))
    evolve(harness, supervised, lineage, max_rounds=2, patience=0)
    assert supervised.interventions == []


def test_the_supervisor_does_not_touch_the_objective(harness: Harness, workdir: Path) -> None:
    """Steering may add a directive; it may never relax what counts as better."""
    lineage = Lineage(workdir / "l.jsonl")
    inner = ScriptedProposer(["score = 1\n", "score = 1.5\n", "score = 2\n"])
    supervised = SupervisedProposer(inner, lineage, Supervisor(stall_after=1))
    result = evolve(harness, supervised, lineage, max_rounds=3, patience=0)
    assert result.kept == 0, "steering changed what counted as an improvement"


# =============================================================================
# Many proposers, one lineage
# =============================================================================
# The intended shape is N cheap proposers ("neurons") working ONE lineage in
# parallel -- which is the point, because a proposer that can see what the
# others already tried is searching rather than guessing. That writes
# concurrently by construction.
#
# MEASURED before the fix, 8 concurrent writers: 1 trial persisted, 7 LOST, and
# 7 of 8 raised PermissionError. The whole-file rewrite was last-writer-wins,
# and on Windows os.replace fails outright when another handle holds the target.


def test_concurrent_proposers_lose_no_trials(tmp_path: Path) -> None:
    """N writers, N trials. None may be lost and none may crash."""
    import threading

    path = tmp_path / "l.jsonl"
    n = 16
    lineages = [Lineage(path) for _ in range(n)]
    errors: list[str] = []

    def work(lineage: Lineage, i: int) -> None:
        try:
            trial = lineage.open_trial(f"neuron {i}")
            lineage.settle(trial, "kept", score=float(i))
        except Exception as exc:  # noqa: BLE001 - the point is to catch ANY
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work, args=(lin, i))
               for i, lin in enumerate(lineages)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = Lineage(path)
    assert not errors, f"writers crashed: {errors[:2]}"
    assert len(final) == n, f"{n - len(final)} trial(s) lost to a concurrent write"
    assert len([t for t in final.trials if t.outcome == "kept"]) == n
    assert sorted(t.score for t in final.trials) == [float(i) for i in range(n)]


def test_a_proposer_sees_trials_written_by_another(tmp_path: Path) -> None:
    """The whole reason to share a lineage: one neuron must see another's work.

    A cached in-memory list would make each proposer blind to its peers, and
    they would re-propose what somebody already measured.
    """
    path = tmp_path / "l.jsonl"
    a, b = Lineage(path), Lineage(path)
    trial = a.open_trial("a's idea")
    a.settle(trial, "reverted", score=1.0, detail="worse")
    assert len(b) == 1, "the second proposer cannot see the first's trial"
    assert "a's idea" in b.render()
    assert "reverted" in b.render()


def test_trial_ids_are_unique_under_contention(tmp_path: Path) -> None:
    """`index` is a display ordinal, not a key: two proposers opening a trial at
    the same instant are handed the same index, so the FILENAME must not use
    it."""
    path = tmp_path / "l.jsonl"
    ids = {Lineage(path).open_trial(f"t{i}").trial_id for i in range(20)}
    assert len(ids) == 20, "trial ids collided; files would overwrite each other"

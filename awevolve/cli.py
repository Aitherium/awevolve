"""awevolve command line.

    awevolve run       --harness h.json [--rounds N] [--agent CMD]
    awevolve lineage   --lineage lineage.jsonl
    awevolve status    --lineage lineage.jsonl
    awevolve check     --harness h.json
    awevolve --self-test

``check`` is separate from ``run`` on purpose: the scorer-boundary refusal and
the availability question are worth answering before spending an evaluation
budget, and a stranger's first command should be one that cannot change
anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .harness import HarnessError, load_harness
from .lineage import Lineage
from .loop import evolve
from .proposer import AGENT_CMD_ENV, AgentProposer
from .scorer import ScoreError
from .supervisor import SupervisedProposer, Supervisor

__all__ = ["main"]


def _emit(event: str, payload: dict) -> None:
    if event == "baseline":
        print(f"baseline {payload['metric']}={payload['score']:g}")
    elif event == "kept":
        print(f"  round {payload['round']}: KEPT   score={payload['score']:g}")
    elif event == "reverted":
        print(f"  round {payload['round']}: revert score={payload['score']:g}")
    elif event == "refused":
        print(f"  round {payload['round']}: refused ({payload.get('reason', '')[:80]})")


def cmd_run(args: argparse.Namespace) -> int:
    harness = load_harness(Path(args.harness))
    available, why = harness.is_available()
    if not available:
        print(f"harness {harness.name!r} is not available: {why}", file=sys.stderr)
        return 2

    lineage_path = Path(args.lineage or f"{harness.name}-lineage.jsonl")
    lineage = Lineage(lineage_path, git_root=args.git_root)

    try:
        agent = AgentProposer(args.agent)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    proposer = SupervisedProposer(
        agent, lineage,
        Supervisor(stall_after=args.stall_after),
        on_intervene=lambda i: print(f"  supervisor: {i.reason}"),
    )

    try:
        result = evolve(
            harness, proposer, lineage,
            max_rounds=args.rounds,
            patience=args.patience,
            on_event=_emit,
        )
    except ScoreError as exc:
        print(f"could not establish a baseline: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2))
    print(f"lineage: {lineage_path}")
    # A run that kept nothing is not a failure -- it is a real, reportable
    # result. Exit 0 and let the caller read `kept`.
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    lineage = Lineage(Path(args.lineage))
    if not len(lineage):
        print("(empty lineage)")
        return 0
    print(lineage.render(limit=args.limit))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    lineage = Lineage(Path(args.lineage))
    print(json.dumps(lineage.summary(), indent=2))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Validate a harness without running anything."""
    try:
        harness = load_harness(Path(args.harness))
    except HarnessError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    available, why = harness.is_available()
    print(json.dumps({
        "name": harness.name,
        "available": available,
        "reason": why,
        "metric": harness.metric_name,
        "direction": "minimize" if harness.minimize else "maximize",
        "confirm_repeats": harness.confirm_repeats,
        "scorer_boundary": "ok (the mutable file is not its own scorer)",
    }, indent=2))
    return 0 if available else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="awevolve", description=__doc__)
    p.add_argument("--self-test", action="store_true",
                   help="prove the invariants still hold")
    sub = p.add_subparsers(dest="command")

    run = sub.add_parser("run", help="evolve a harness")
    run.add_argument("--harness", required=True)
    run.add_argument("--lineage", default="")
    run.add_argument("--agent", default="",
                     help=f"agent command (or set {AGENT_CMD_ENV})")
    run.add_argument("--rounds", type=int, default=10)
    run.add_argument("--patience", type=int, default=3)
    run.add_argument("--stall-after", type=int, default=3)
    run.add_argument("--git-root", default="")
    run.set_defaults(func=cmd_run)

    lin = sub.add_parser("lineage", help="show what has been tried")
    lin.add_argument("--lineage", required=True)
    lin.add_argument("--limit", type=int, default=50)
    lin.set_defaults(func=cmd_lineage)

    st = sub.add_parser("status", help="summarise a lineage")
    st.add_argument("--lineage", required=True)
    st.set_defaults(func=cmd_status)

    ck = sub.add_parser("check", help="validate a harness, change nothing")
    ck.add_argument("--harness", required=True)
    ck.set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


def self_test() -> int:
    """Prove the invariants can still fail.

    Runs the real loop against a scorer that is a deterministic function of the
    file's contents, so keep/revert decisions are reproducible rather than
    depending on a live benchmark.
    """
    import tempfile

    from .harness import Harness, SelfScoringError
    from .proposer import ScriptedProposer

    results: dict[str, bool] = {}

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "config.py").write_text("score = 5\n", encoding="utf-8")
        (d / "score.py").write_text(
            "import io, re\n"
            "s = io.open('config.py', encoding='utf-8').read()\n"
            "m = re.search(r'score\\s*=\\s*([0-9.]+)', s)\n"
            "print('METRIC v=' + (m.group(1) if m else '0'))\n",
            encoding="utf-8",
        )

        def _harness(**kw):
            opts = dict(
                name="t", mutable_file=d / "config.py",
                # sys.executable, never the bare `python` name: Debian ships
                # python3 only, and a scorer that dies `python: not found`
                # (exit 127) refuses every candidate — measured 2026-08-26 by
                # the sync lane's own test step, which caught it pre-mirror.
                eval_command=f"{sys.executable} score.py",
                metric_regex=r"METRIC v=([0-9.]+)",
                metric_name="v", minimize=False, base_dir=d, time_budget_s=60,
            )
            opts.update(kw)
            return Harness(**opts)

        # a winning change reaches disk
        h = _harness()
        lin = Lineage(d / "a.jsonl")
        r = evolve(h, ScriptedProposer(["score = 9\n"]), lin, max_rounds=1, patience=0)
        results["a winning change is kept on disk"] = (
            r.kept == 1 and "score = 9" in (d / "config.py").read_text(encoding="utf-8")
        )

        # a losing change is restored
        (d / "config.py").write_text("score = 5\n", encoding="utf-8")
        lin = Lineage(d / "b.jsonl")
        r = evolve(h, ScriptedProposer(["score = 1\n"]), lin, max_rounds=1, patience=0)
        results["a losing change is restored"] = (
            r.reverted == 1
            and (d / "config.py").read_text(encoding="utf-8") == "score = 5\n"
        )

        # a no-op is refused, never scored
        lin = Lineage(d / "c.jsonl")
        r = evolve(h, ScriptedProposer(["score = 5\n"]), lin, max_rounds=1, patience=0)
        results["a no-op is refused, not scored"] = (
            r.refused == 1 and r.kept == 0 and r.reverted == 0
        )

        # the proposer is handed the lineage
        sp = ScriptedProposer(["score = 6\n", "score = 7\n"])
        lin = Lineage(d / "d.jsonl")
        (d / "config.py").write_text("score = 5\n", encoding="utf-8")
        evolve(h, sp, lin, max_rounds=2, patience=0)
        results["the proposer receives the lineage"] = (
            len(sp.seen) >= 2 and "lineage" in sp.seen[1] and sp.seen[1]["lineage"]
        )

        # a non-zero exit refuses the trial
        (d / "config.py").write_text("score = 5\n", encoding="utf-8")
        strict = _harness(eval_command=f"{sys.executable} score.py && exit 1")
        lin = Lineage(d / "e.jsonl")
        try:
            r = evolve(strict, ScriptedProposer(["score = 9\n"]), lin,
                       max_rounds=1, patience=0, baseline=5.0)
            results["a non-zero exit refuses the trial"] = r.refused == 1
        except ScoreError:
            results["a non-zero exit refuses the trial"] = True

        # the scorer boundary is enforced at construction
        try:
            _harness(mutable_file=d / "score.py")
            results["a self-scoring harness is refused"] = False
        except SelfScoringError:
            results["a self-scoring harness is refused"] = True

        # a bad metric regex is refused
        try:
            _harness(metric_regex=r"METRIC v=[0-9.]+")  # no capture group
            results["a regex with no capture group is refused"] = False
        except HarnessError:
            results["a regex with no capture group is refused"] = True

    width = max(len(k) for k in results)
    for label, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<{width}}")
    failed = [k for k, v in results.items() if not v]
    print()
    if failed:
        print(f"SELF-TEST FAILED: {len(failed)} arm(s).")
        return 1
    print(f"SELF-TEST OK: {len(results)} arms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

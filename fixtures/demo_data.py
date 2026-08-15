#!/usr/bin/env python3
"""A larger, more varied synthetic dataset — for screenshots and local
demoing only, never for tests.

`generate.py`'s fixtures are deliberately small and exact: each one exists to
exercise one detector precisely, and the test suite asserts precise counts
against them. This file is the opposite job — it exists to make the dashboard
look like something a real person would actually have accumulated over a
couple of months, so screenshots aren't obviously a toy. Still 100% synthetic
(see the project-wide rule: no real session data is ever committed).

    python fixtures/demo_data.py DIR   # writes DIR/projects/...

Point `CLAUDE_CONFIG_DIR` at the parent of DIR and `agentlens ingest`.
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import MODELS, Writer  # noqa: E402

PROJECTS = [
    "/home/dev/checkout-service",
    "/home/dev/web-app",
    "/home/dev/api-gateway",
    "/home/dev/billing",
    "/home/dev/payments-api",
    "/home/dev/data-pipeline",
]


def _sid() -> str:
    return str(uuid.uuid4())


def healthy(rng: random.Random, cwd: str, start: datetime, task: str, path: str, test_cmd: str) -> Writer:
    """A well-run session — most of a real history looks like this, not like
    a finding. The dashboard needs to prove it stays quiet just as often as
    it needs to prove it speaks up."""
    w = Writer(_sid(), cwd, start)
    model = rng.choice(MODELS)
    w.summary(task)
    w.user(f"In @{path} {task.lower()}; `{test_cmd}` must pass.")
    ids = w.assistant("On it.", model, rng.randint(15000, 45000), rng.randint(1200, 3200), [
        ("Read", {"file_path": f"{cwd}/{path}"}),
        ("Edit", {"file_path": f"{cwd}/{path}"}),
        ("Bash", {"command": test_cmd}),
    ])
    w.tool_result(ids[0], "def existing(): pass\n")
    w.tool_result(ids[1], "ok")
    w.tool_result(ids[2], f"{rng.randint(2, 12)} passed in {rng.uniform(0.2, 1.4):.1f}s")
    w.assistant("Done — tests pass.", model, rng.randint(30000, 70000), rng.randint(400, 900))
    return w


def vague(rng: random.Random, cwd: str, start: datetime, area: str, path: str) -> Writer:
    w = Writer(_sid(), cwd, start)
    model = rng.choice(MODELS)
    w.summary(f"Clean up {area}")
    w.user(f"clean up the {area} stuff")
    ids = w.assistant("Looking around.", model, rng.randint(12000, 22000), 700, [
        ("Glob", {"pattern": f"**/*{area}*"}),
        ("Grep", {"pattern": area}),
        ("Read", {"file_path": f"{cwd}/{path}"}),
    ])
    w.tool_result(ids[0], f"{path}\n")
    w.tool_result(ids[1], f"{path}:20\n{path}:55")
    w.tool_result(ids[2], "x = 1\n" * rng.randint(300, 900))
    for _ in range(rng.randint(2, 4)):
        i2 = w.assistant("Re-checking the file.", model, rng.randint(28000, 45000), 900,
                          [("Read", {"file_path": f"{cwd}/{path}"})])
        w.tool_result(i2[0], "x = 1\n" * 900)
    for _ in range(rng.randint(4, 9)):
        w.assistant("Editing.", model, rng.randint(35000, 60000), rng.randint(900, 1800),
                    [("Edit", {"file_path": f"{cwd}/{path}"})])
    return w


def output_flood(rng: random.Random, cwd: str, start: datetime, cmd: str) -> Writer:
    w = Writer(_sid(), cwd, start)
    model = rng.choice(MODELS)
    w.summary("Audit dependency tree")
    w.user(f"check our dependencies with `{cmd}` for anything odd")
    ids = w.assistant("Checking.", model, rng.randint(8000, 15000), 500, [("Bash", {"command": cmd})])
    w.tool_result(ids[0], "output line\n" * rng.randint(2500, 5000))
    for i in range(rng.randint(3, 6)):
        w.assistant(f"Analysing part {i}.", model, 60000 + i * 15000, rng.randint(600, 1400))
    return w


def blind_retry(rng: random.Random, cwd: str, start: datetime, thing: str, path: str) -> Writer:
    w = Writer(_sid(), cwd, start)
    model = rng.choice(MODELS)
    w.summary(f"Fix the failing {thing}")
    w.user(f"the {thing} is broken, can you fix it")
    ids = w.assistant("Let me look.", model, rng.randint(10000, 20000), 900,
                       [("Grep", {"pattern": thing}), ("Read", {"file_path": f"{cwd}/{path}"})])
    w.tool_result(ids[0], f"{path}:12\n")
    w.tool_result(ids[1], "def handle(req):\n" + "    ...\n" * 50)
    w.assistant("Changed it.", model, rng.randint(25000, 40000), rng.randint(1200, 2200),
                [("Edit", {"file_path": f"{cwd}/{path}"})])
    w.user("still not working, please fix it")
    w.assistant("Trying again.", model, rng.randint(40000, 60000), rng.randint(1200, 2200),
                [("Edit", {"file_path": f"{cwd}/{path}"})])
    w.user("that didn't work either")
    w.assistant("Once more.", model, rng.randint(55000, 75000), rng.randint(1200, 2200),
                [("Edit", {"file_path": f"{cwd}/{path}"})])
    w.user(f"I ran the tests and got: AssertionError: expected 200 got 401 at {path.split('/')[-1]} "
           "line 42. Expected the handler to accept a valid signature.")
    w.assistant("That's the header casing. Fixed.", model, rng.randint(65000, 85000), rng.randint(600, 1000),
                [("Edit", {"file_path": f"{cwd}/{path}"})])
    return w


def loop(rng: random.Random, cwd: str, start: datetime, path: str, test_cmd: str,
         exc: str, n: int = 4) -> Writer:
    """A real, specific failure (not a generic one — see the loop-cost bug
    fixed 2026-08-15: overly generic signatures falsely match across
    sessions and inflate the dollar total past what was actually spent)."""
    w = Writer(_sid(), cwd, start)
    model = rng.choice(MODELS)
    w.summary(f"Fix {exc.split(':')[0]}")
    w.user(f"In @{path} fix this properly; `{test_cmd}` must pass.")
    for i in range(n):
        ids = w.assistant(f"Attempt {i + 1}.", model, 22000 + i * 12000, rng.randint(900, 1800),
                           [("Edit", {"file_path": f"{cwd}/{path}"}), ("Bash", {"command": test_cmd})])
        w.tool_result(ids[0], "ok")
        w.tool_fail(ids[1],
                     f'Traceback (most recent call last):\n  File "{cwd}/{path}", line {40 + i * 3}, '
                     f"in test_case\n    assert result is True\n{exc} (ran in 0.{i}3s)")
    return w


def build(target: Path, weeks: int = 14, sessions_per_week: tuple[int, int] = (4, 8)) -> list[Path]:
    rng = random.Random(42)
    base = datetime(2026, 6, 1, 9, 0, 0)
    written: list[Path] = []

    tasks = [
        ("add pagination cursor support", "src/api/pages.py", "pytest tests/test_pages.py -q"),
        ("add retry backoff to the queue consumer", "src/queue/consumer.py", "pytest tests/test_consumer.py -q"),
        ("add rate limiting to the public endpoint", "src/api/public.py", "pytest tests/test_public.py -q"),
        ("normalize currency formatting", "src/billing/format.py", "pytest tests/test_format.py -q"),
        ("add idempotency keys to the charge endpoint", "src/payments/charge.py", "pytest tests/test_charge.py -q"),
        ("fix timezone handling in the scheduler", "src/pipeline/schedule.py", "pytest tests/test_schedule.py -q"),
    ]
    vague_areas = [("auth", "src/auth.py"), ("caching", "src/cache.py"), ("logging", "src/log.py")]
    flood_cmds = ["npm ls --all", "pip freeze", "docker images -a", "find . -name '*.py'"]
    retry_things = [("webhook handler", "src/webhook.py"), ("email sender", "src/notify/email.py")]

    day = 0
    for _week in range(weeks):
        n = rng.randint(*sessions_per_week)
        for _ in range(n):
            day_offset = day + rng.randint(0, 6)
            hour = rng.randint(8, 19)
            start = base + timedelta(days=day_offset, hours=hour, minutes=rng.randint(0, 59))
            cwd = rng.choice(PROJECTS)
            kind = rng.choices(
                ["healthy", "vague", "flood", "retry"],
                weights=[62, 15, 10, 13],
            )[0]
            if kind == "healthy":
                task, path, cmd = rng.choice(tasks)
                w = healthy(rng, cwd, start, task, path, cmd)
            elif kind == "vague":
                area, path = rng.choice(vague_areas)
                w = vague(rng, cwd, start, area, path)
            elif kind == "flood":
                w = output_flood(rng, cwd, start, rng.choice(flood_cmds))
            else:
                thing, path = rng.choice(retry_things)
                w = blind_retry(rng, cwd, start, thing, path)
            enc = w.cwd.replace("/", "-")
            p = target / enc / f"{w.sid}.jsonl"
            w.write(p)
            written.append(p)
        day += 7

    # A handful of real, specific loops — enough to make the Engineering and
    # Loops story worth looking at without being the $19k-style bug this
    # generator exists partly to guard against reproducing visually.
    loop_specs = [
        ("/home/dev/payments-api", "src/payments/verify.py", "pytest tests/test_verify.py -q",
         "AssertionError: expected True, got False"),
        ("/home/dev/api-gateway", "src/auth/token.py", "pytest tests/test_token.py -q",
         "AssertionError: expected 'valid', got 'expired'"),
        ("/home/dev/data-pipeline", "src/pipeline/retry.py", "pytest tests/test_retry.py -q",
         "AssertionError: expected 3 retries, got 1"),
    ]
    for i, (cwd, path, cmd, exc) in enumerate(loop_specs):
        start = base + timedelta(days=10 + i * 20, hours=10)
        w = loop(rng, cwd, start, path, cmd, exc, n=rng.randint(3, 5))
        p = target / w.cwd.replace("/", "-") / f"{w.sid}.jsonl"
        w.write(p)
        written.append(p)
        # Recur once, a few weeks later, in a fresh session — what makes
        # CrossSessionLoopDetector fire, and realistically (once, not
        # dozens of times against unrelated sessions).
        start2 = start + timedelta(days=rng.randint(14, 28))
        w2 = loop(rng, cwd, start2, path, cmd, exc, n=rng.randint(2, 3))
        p2 = target / w2.cwd.replace("/", "-") / f"{w2.sid}.jsonl"
        w2.write(p2)
        written.append(p2)

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="directory to write projects/ under")
    ap.add_argument("--weeks", type=int, default=9)
    args = ap.parse_args()
    paths = build(args.target / "projects", weeks=args.weeks)
    print(f"Wrote {len(paths)} synthetic sessions under {args.target / 'projects'}")


if __name__ == "__main__":
    main()

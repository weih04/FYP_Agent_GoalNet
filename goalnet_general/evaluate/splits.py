"""Development and test tasks, and the prompt freeze between them.

Prompts are tuned on an environment's development tasks only (its adapter's
`tasks["dev"]`). When they are final,

    python -m goalnet_general.evaluate.splits freeze

records prompts.fingerprint() in FROZEN_PROMPT.json. The prompts are environment-
agnostic, so one freeze covers every environment. From then on a run on a test
task must use exactly that prompt: run_construct refuses otherwise, unless
--unfrozen-ok is given, in which case the run is marked prompt_frozen=false and
aggregate.py leaves it out of test results. Editing prompts.py, schema.py,
problems.py or evaluate/metrics/validity.py changes the fingerprint, so any later tuning is visible
rather than silent.
"""

import datetime
import json
import os
import sys

from ..prompts import fingerprint

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "FROZEN_PROMPT.json")


def split_of(task, adapter):
    for split in ("dev", "test"):
        if task in adapter.tasks[split]:
            return split
    raise ValueError(f"{adapter.name} task {task!r} is in neither split; add it to the adapter's tasks")


def frozen_fingerprint():
    if not os.path.exists(FROZEN):
        return None
    with open(FROZEN) as f:
        return json.load(f)["fingerprint"]


def check(task, adapter, unfrozen_ok=False):
    """{split, prompt_fingerprint, prompt_frozen} for a run; raises on an unfrozen test run."""
    split, current, frozen = split_of(task, adapter), fingerprint(), frozen_fingerprint()
    is_frozen = frozen is not None and current == frozen
    if split == "test" and not is_frozen and not unfrozen_ok:
        state = "no prompt has been frozen" if frozen is None else f"frozen prompt is {frozen}"
        raise RuntimeError(
            f"{task} is a test task, but the current prompt {current} is not frozen ({state}). "
            "Finish tuning on the dev tasks and run `python -m goalnet_general.evaluate.splits freeze`, or pass "
            "--unfrozen-ok to run anyway (the run is then excluded from test results).")
    return {"split": split, "prompt_fingerprint": current, "prompt_frozen": is_frozen}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["freeze"]:
        record = {"fingerprint": fingerprint(),
                  "frozen_at": datetime.datetime.now().isoformat(timespec="seconds")}
        with open(FROZEN, "w") as f:
            json.dump(record, f, indent=2)
        print(f"froze prompt {record['fingerprint']} in {FROZEN}")
        return 0
    print(f"current prompt {fingerprint()}, frozen {frozen_fingerprint()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

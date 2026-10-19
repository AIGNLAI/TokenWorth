"""JSON-lines worker for isolated, persistent BigCodeBench evaluation."""

from __future__ import annotations

import contextlib
import json
import sys

from bigcodebench.data import get_bigcodebench
from bigcodebench.eval import untrusted_check


def main() -> None:
    problems = get_bigcodebench(subset="full")
    print(json.dumps({"ready": True, "tasks": len(problems)}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            task = problems[request["task_id"]]
            calibrated = task["code_prompt"] + "\n    pass\n" + request["solution"]
            # BigCodeBench's safety wrappers print intercepted subprocess
            # diagnostics.  Keep those (and output inherited by forked test
            # processes) off stdout, which is reserved for the JSONL protocol.
            with contextlib.redirect_stdout(sys.stderr):
                status, details = untrusted_check(
                    calibrated,
                    task["test"],
                    task["entry_point"],
                    request.get("max_as_limit", 30 * 1024),
                    request.get("max_data_limit", 30 * 1024),
                    request.get("max_stack_limit", 10),
                    request.get("min_time_limit", 1),
                    request.get("gt_time_limit", 20),
                )
            response = {"status": status or "timeout", "details": details}
        except BaseException as error:
            response = {
                "status": "eval_error",
                "details": {"worker": f"{type(error).__name__}: {error}"},
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

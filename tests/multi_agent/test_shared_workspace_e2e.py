"""Exercise shared state, task scoring, and HTML output through the example."""

import json
import subprocess
import sys
from pathlib import Path


def test_parallel_agents_share_state_and_sequential_agents_do_not(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    results = {}
    for mode, flags in (("parallel", []), ("sequential", ["--sequential"])):
        output = tmp_path / mode
        subprocess.run(
            [
                sys.executable,
                str(root / "examples/multi-agent/run.py"),
                "--sandbox",
                "local",
                "--out-dir",
                str(output),
                *flags,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        results[mode] = json.loads((output / "summary.json").read_text())

    parallel = results["parallel"]
    assert parallel["model_calls"] == 0
    assert parallel["scores"]["inventory"] == "C"
    assert parallel["scores"]["schedule"] == "C"
    assert parallel["scores"]["tasks"]["main"] == 1.0
    assert max(item["ready_at"] for item in parallel["evidence"].values()) < min(
        item["finished_at"] for item in parallel["evidence"].values()
    )
    assert Path(parallel["log"]).is_file()
    assert all(Path(page).is_file() for page in parallel["html"])

    sequential = results["sequential"]
    assert sequential["scores"]["inventory"] == "I"
    assert sequential["scores"]["tasks"]["main"] == 0.5
    assert (
        sequential["evidence"]["inventory"]["finished_at"]
        < sequential["evidence"]["schedule"]["ready_at"]
    )

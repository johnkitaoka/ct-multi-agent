"""Run two benign MiniLab operators in one sandbox and render their eval log."""

import argparse
import json
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, execute_tools
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolCall, ToolDef
from inspect_ai.util import sandbox

from control_tower.ct_runs.ai_eval.dataset import to_dataset
from control_tower.environments.all_environments import get_environment
from control_tower.eval_logs.html.render import render_eval_files
from control_tower.multi_agent.runner import AgentRun, run_agents
from control_tower.sandbox.cli import SandboxConfig
from control_tower.task.sample_metadata import SampleMetadata
from control_tower.task.scorers import task_scorers
from control_tower.task.setup import prepare_sample
from control_tower.task.space import TaskCombination
from control_tower.task.task import get_full_main_task, get_full_side_task


def operator(name: str, user: str, commands: tuple[str, ...]) -> Agent:
    async def run_command(command: str) -> str:
        """Run one assigned operator command.

        Args:
            command: The operator command to run.
        """
        result = await sandbox().exec(["bash", "-lc", command], user=user)
        if not result.success:
            raise RuntimeError(f"{name}: {command}: {result.stderr or result.stdout}")
        return result.stdout

    tool = ToolDef(run_command, name="run_operator_command").as_tool()

    async def work(state: AgentState) -> AgentState:
        for index, command in enumerate(commands):
            state.messages.append(
                ChatMessageAssistant(
                    content=f"{name}: {command}",
                    tool_calls=[
                        ToolCall(
                            id=f"{name}-{index}",
                            function="run_operator_command",
                            arguments={"command": command},
                        )
                    ],
                )
            )
            result = await execute_tools(state.messages, [tool])
            state.messages.extend(result.messages)
        state.messages.append(ChatMessageAssistant(content=f"{name} completed."))
        return state

    return work


@solver
def roster() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        metadata = state.metadata_as(SampleMetadata)
        mains = [
            get_full_main_task(data, metadata.environment.env_args)
            for data in metadata.main_tasks
        ]
        sides = [
            get_full_side_task(data, metadata.environment.env_args)
            for data in metadata.side_tasks
        ]
        await prepare_sample([*mains, *sides])
        results = await run_agents(
            [
                AgentRun(
                    "window",
                    operator("window", "svc-window", ("renew-window",)),
                    [ChatMessageUser(content=mains[0].description)],
                ),
                AgentRun(
                    "package",
                    operator(
                        "package",
                        "svc-deps",
                        ("allow-package", "allow-package --package scipy"),
                    ),
                    [ChatMessageUser(content=mains[1].description)],
                ),
            ]
        )
        state.messages = [
            message for result in results.values() for message in result.messages
        ]
        return state

    return solve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--with-side", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Choose a new output directory to preserve prior recordings")
    environment = get_environment("minilab")
    space = environment.task_space()
    combination = TaskCombination(
        mains=(
            space.mains()["fl_train_renew_window"],
            space.mains()["fl_pf_allow_package"],
        ),
        sides=(space.sides()["fl_pf_blinded_watch"],) if args.with_side else (),
    )
    dataset = to_dataset(
        [combination], sandbox_config=SandboxConfig(allow_internet=True)
    )
    logs = inspect_eval(
        Task(
            name="minilab_shared_operators",
            dataset=dataset,
            solver=roster(),
            scorer=task_scorers(dataset, score_main_on_attack=True),
        ),
        model="none",
        log_dir=str(args.out_dir / "logs"),
        display="none",
        max_samples=1,
    )
    log = logs[0]
    if log.status != "success" or not log.location:
        raise RuntimeError(
            f"MiniLab evaluation failed: {log.error.message if log.error else log.status}"
        )
    pages = render_eval_files([log.location], args.out_dir / "html")
    sample = (log.samples or [])[0]
    summary = {
        "log": str(Path(log.location).resolve()),
        "html": [str(page.resolve()) for page in pages],
        "scores": {name: score.value for name, score in (sample.scores or {}).items()},
        "model_calls": sum(event.event == "model" for event in sample.events),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

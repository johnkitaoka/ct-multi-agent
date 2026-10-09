"""Run a benign shared-sandbox roster and render its real Inspect log."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from demo_environment import environment
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, execute_tools
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolCall, ToolDef
from inspect_ai.util import sandbox, store_as

from control_tower.ct_runs.ai_eval.dataset import to_dataset
from control_tower.eval_logs.html.render import render_eval_files
from control_tower.models.model_spec import ModelSpec
from control_tower.multi_agent.runner import (
    AgentRun,
    MultiAgentTranscripts,
    run_agents,
)
from control_tower.multi_agent.scaffold import (
    AgentConfig,
    mailbox_tools,
    register_tool_group,
)
from control_tower.multi_agent.solver import MultiAgentArgs, multi_agent_solver
from control_tower.sandbox.cli import SandboxConfig
from control_tower.task.sample_metadata import SampleMetadata
from control_tower.task.scorers import task_scorers
from control_tower.task.setup import prepare_sample
from control_tower.task.space import TaskCombination
from control_tower.task.task import get_full_main_task


def workspace_tools(name: str, peer: str) -> list[Tool]:
    async def publish_note() -> str:
        """Write a readiness note in the shared directory."""
        timestamp = time.time()
        await sandbox().write_file(f"{name}.ready", str(timestamp))
        return json.dumps({"agent": name, "ready_at": timestamp})

    async def wait_for_peer() -> str:
        """Wait up to two seconds for the other worker's note."""
        deadline = time.monotonic() + 2
        peer_ready = False
        while time.monotonic() < deadline:
            try:
                await sandbox().read_file(f"{peer}.ready")
            except FileNotFoundError:
                await asyncio.sleep(0.02)
            else:
                peer_ready = True
                break
        await sandbox().write_file(f"{name}.observed", json.dumps(peer_ready))
        return json.dumps({"peer": peer, "peer_ready": peer_ready})

    async def write_report() -> str:
        """Write the assigned report using setup inputs and the peer's readiness."""
        report = {
            "agent": name,
            "input": (await sandbox().read_file(f"{name}.input")).strip(),
            "setup": await sandbox().read_file(f"{name}.setup"),
            "peer_ready": json.loads(await sandbox().read_file(f"{name}.observed")),
            "finished_at": time.time(),
        }
        text = json.dumps(report)
        await sandbox().write_file(f"{name}.report.json", text)
        return text

    return [ToolDef(fn).as_tool() for fn in (publish_note, wait_for_peer, write_report)]


def scripted_worker(name: str, peer: str, max_steps: int) -> Agent:
    tools = workspace_tools(name, peer) + mailbox_tools(name, [name, peer])

    async def work(state: AgentState) -> AgentState:
        steps = [
            [
                ("publish_note", {}),
                ("send_message", {"to_agent": peer, "content": f"{name} is ready"}),
            ],
            [("wait_for_peer", {})],
            [("write_report", {}), ("read_messages", {})],
        ]
        for step, calls in enumerate(steps, start=1):
            state.messages.append(
                ChatMessageAssistant(
                    content=f"{name}: step {step}",
                    tool_calls=[
                        ToolCall(id=f"{name}-{step}-{i}", function=fn, arguments=args)
                        for i, (fn, args) in enumerate(calls)
                    ],
                )
            )
            result = await execute_tools(state.messages, tools)
            state.messages.extend(result.messages)
            if step >= max_steps:
                break
        state.messages.append(
            ChatMessageAssistant(content=f"{name} finished its allotted steps.")
        )
        return state

    return work


@solver
def demo_solver(parallel: bool = True, max_steps: int = 3) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        metadata = state.metadata_as(SampleMetadata)
        await prepare_sample(
            [
                get_full_main_task(task, metadata.environment.env_args)
                for task in metadata.main_tasks
            ]
        )
        runs = [
            AgentRun(
                name,
                scripted_worker(name, peer, max_steps),
                [ChatMessageUser(content=f"Private assignment for {name}")],
            )
            for name, peer in (("inventory", "schedule"), ("schedule", "inventory"))
        ]
        results = await run_agents(runs, parallel=parallel)
        state.messages = [
            message for result in results.values() for message in result.messages
        ]
        state.output = results["schedule"].output
        state.metadata["agent_outcomes"] = store_as(MultiAgentTranscripts).outcomes
        evidence: dict[str, dict[str, object]] = {}
        for name in ("inventory", "schedule"):
            ready = float(await sandbox().read_file(f"{name}.ready"))
            try:
                report = json.loads(await sandbox().read_file(f"{name}.report.json"))
            except FileNotFoundError:
                report = {}
            evidence[name] = {"ready_at": ready, **report}
        state.metadata["shared_workspace_evidence"] = evidence
        return state

    return solve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sandbox", choices=("docker", "local"), default="docker")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument(
        "--model",
        help="Use model-driven agents instead of scripted operators (requires API access)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved task and policy settings without running agents",
    )
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.out_dir.exists():
        parser.error("Choose a new output directory to preserve prior recordings")
    tasks = tuple(environment.task_space().mains().values())
    dataset = to_dataset(
        [TaskCombination(mains=tasks)],
        sandbox_config=SandboxConfig(allow_internet=args.sandbox == "local"),
        env_args={environment.id: {"sandbox_kind": args.sandbox}},
    )
    policy_args = MultiAgentArgs(
        model=ModelSpec(model_name=args.model or "none"),
        agents=[
            AgentConfig(agent_id=name, main_task_id=name, tool_group=name)
            for name in ("inventory", "schedule")
        ],
        parallel=not args.sequential,
        comms_enabled=True,
        max_steps=args.max_steps,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "sample": dataset[0].metadata,
                    "policy": policy_args.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return
    if args.model and args.sandbox == "local":
        parser.error("Model-driven agents require --sandbox docker")
    if args.model:
        register_tool_group(
            "inventory", lambda: workspace_tools("inventory", "schedule")
        )
        register_tool_group(
            "schedule", lambda: workspace_tools("schedule", "inventory")
        )
        main_solver = multi_agent_solver(policy_args)
    else:
        main_solver = demo_solver(not args.sequential, args.max_steps)
    task = Task(
        name="shared_workspace",
        dataset=dataset,
        solver=main_solver,
        scorer=task_scorers(dataset),
    )
    logs = inspect_eval(
        task,
        model="none",
        log_dir=str(args.out_dir / "logs"),
        display="none",
        max_samples=1,
    )
    log = logs[0]
    if log.status != "success" or not log.location:
        raise RuntimeError(
            f"Demo evaluation failed: {log.error.message if log.error else log.status}"
        )
    pages = render_eval_files([log.location], args.out_dir / "html")
    sample = (log.samples or [])[0]
    summary = {
        "log": str(Path(log.location).resolve()),
        "html": [str(page.resolve()) for page in pages],
        "scores": {key: value.value for key, value in (sample.scores or {}).items()},
        "evidence": sample.metadata.get("shared_workspace_evidence", {}),
        "model_calls": sum(event.event == "model" for event in sample.events),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

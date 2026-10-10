"""Wire a red policy + blue protocol into a per-sample solver."""

from copy import copy
from dataclasses import replace

from inspect_ai.agent import AgentState
from inspect_ai.solver import Generate, Solver, TaskState, solver

from control_tower.multi_agent.runner import AgentRun, run_agents
from control_tower.multi_agent.scaffold import mailbox_tools
from control_tower.protocols.eval.blue import Game, Protocol
from control_tower.protocols.eval.game_view import RedView
from control_tower.protocols.eval.harness import (
    InspectSweHarness,
    NullHarness,
    RosterHarness,
)
from control_tower.protocols.eval.red import Policy, Untrusted
from control_tower.protocols.eval.registry import (
    PolicyFactory,
    ProtocolFactory,
    policies,
)
from control_tower.protocols.eval.white import White
from control_tower.sandbox.replay import active_replay_sandbox
from control_tower.task.sample_metadata import SampleMetadata, Stream
from control_tower.task.setup import prepare_sample
from control_tower.task.task import get_full_main_task, get_full_side_task
from control_tower.tools.linux_tools import TASK_TEST_TOOLS, build_sample_tools
from control_tower.tools.registry import ensure_unique_tool_keys, tool_key
from control_tower.utils.errors import ValidationError
from control_tower.utils.function_registry.base_model import BaseModelPython
from control_tower.utils.function_registry.function_info import FunctionInfo


def build_eval_solver(
    policy_info: "FunctionInfo[PolicyFactory]",
    policy_args: BaseModelPython | None,
    protocol_info: "FunctionInfo[ProtocolFactory]",
    protocol_args: BaseModelPython | None,
    *,
    white: White | None = None,
    attack: "tuple[FunctionInfo[PolicyFactory], BaseModelPython | None] | None" = None,
) -> Solver:
    """The red policy acts as U; the blue protocol decides each action; the runner drives. Each team
    chooses its affordances in its own args, and white checks them before the run starts. An
    `attack` policy acts as U on the attack samples, leaving the honest ones to `policy_info`."""
    white = white or White()
    honest = (policy_info, policy_args)
    playing: dict[Stream, tuple[FunctionInfo[PolicyFactory], BaseModelPython | None]]
    playing = {"honest": honest, "attack": attack or honest}
    for team, info, args in (
        ("policy", *honest),
        *([("policy", *attack)] if attack else []),
        ("protocol", protocol_info, protocol_args),
    ):
        if info.args_class is not None:
            white.check(
                f"{team} {info.name!r}", args or info.args_class.model_construct()
            )
    if isinstance(getattr(protocol_args, "harness", None), InspectSweHarness) and (
        scripted := sorted(
            {
                info.name
                for info, _ in playing.values()
                if policies.get_metadata(info.name).get("scripted")
            }
        )
    ):
        raise ValidationError(
            f"scripted policy {scripted} plays its own `submit`, which only the null harness "
            "takes; drop the protocol's `harness` or play a model policy"
        )
    protocol_kwargs = dict(protocol_args) if protocol_args else {}
    allowed_tools = white.private_tools
    roster = getattr(protocol_args, "harness", None)
    if isinstance(roster, RosterHarness) and protocol_info.name != "null-blue-team":
        raise ValidationError("roster harness currently requires null-blue-team")

    @solver
    def eval_runner() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            del generate
            meta = SampleMetadata.model_validate(state.metadata)
            env_args = meta.environment.env_args
            main_tasks = [
                get_full_main_task(data, env_args) for data in meta.main_tasks
            ]
            side_tasks = [
                get_full_side_task(data, env_args) for data in meta.side_tasks
            ]
            playing_info, playing_args = playing[meta.stream]
            if isinstance(roster, RosterHarness):
                ids = [worker.agent_id for worker in roster.workers]
                assigned_ids = [worker.main_task_id for worker in roster.workers]
                main_by_id = {task.id: task for task in main_tasks}
                data_by_id = {task.id: task for task in meta.main_tasks}
                if len(set(ids)) != len(ids):
                    raise ValidationError("roster agent IDs must be unique")
                if len(set(assigned_ids)) != len(assigned_ids) or set(
                    assigned_ids
                ) != set(main_by_id):
                    raise ValidationError(
                        "roster must assign every main task to one worker"
                    )
                await prepare_sample([*main_tasks, *side_tasks])
                state.metadata["agent_ids"] = ids
                state.metadata["agent_assignments"] = dict(
                    zip(ids, assigned_ids, strict=True)
                )
                state.metadata["agent_users"] = {
                    worker.agent_id: worker.user for worker in roster.workers
                }
                runs: list[AgentRun] = []
                games: list[Game] = []
                for worker in roster.workers:
                    worker_state = copy(state)
                    worker_state.metadata = meta.model_copy(
                        update={"main_tasks": (data_by_id[worker.main_task_id],)}
                    ).model_dump(mode="json")
                    worker_state.messages = []
                    tools = build_sample_tools(
                        worker_state,
                        [main_by_id[worker.main_task_id]],
                        side_tasks,
                        text_editor_config="both" if white.split_editor else "default",
                        user=worker.user,
                    )
                    if roster.comms_enabled:
                        offered = [*tools.tools, *mailbox_tools(worker.agent_id, ids)]
                        ensure_unique_tool_keys(offered)
                        tools = replace(tools, tools=offered)
                    policy = playing_info.function(
                        **(dict(playing_args) if playing_args else {})
                    )
                    if (
                        missing := policy.private_tools
                        - set(tools.private_tools)
                        - TASK_TEST_TOOLS
                    ):
                        raise ValidationError(
                            f"policy {playing_info.name!r} uses private tool(s) {sorted(missing)} "
                            f"that worker {worker.agent_id!r} does not have"
                        )
                    chosen = {
                        name: tool
                        for name, tool in sorted(tools.private_tools.items())
                        if name in policy.private_tools and name in allowed_tools
                    }
                    policy.setup(
                        RedView(
                            meta.environment,
                            (data_by_id[worker.main_task_id],),
                            meta.side_tasks,
                        )
                    )
                    untrusted = Untrusted(policy, side_setters=tools.side_setters)
                    untrusted.private_tools = {**tools.side_setters, **chosen}
                    protocol = protocol_info.function(
                        **{**protocol_kwargs, "harness": NullHarness()}
                    )
                    game = Game(
                        state=worker_state,
                        untrusted=untrusted,
                        white=white.model_copy(
                            update={"max_steps": worker.max_steps or white.max_steps}
                        ),
                        protocol=protocol,
                        tools=tools,
                        agent_id=worker.agent_id,
                    )
                    games.append(game)

                    async def play(
                        agent_state: AgentState, game: Game = game
                    ) -> AgentState:
                        await game.run()
                        agent_state.messages = game.final_conversation
                        return agent_state

                    runs.append(
                        AgentRun(worker.agent_id, play, game.history.task_messages)
                    )
                results = await run_agents(runs, parallel=roster.parallel)
                state.messages = [
                    message
                    for result in results.values()
                    for message in result.messages
                ]
                state.tools = list(
                    {
                        tool_key(tool): tool
                        for game in games
                        for tool in game.history.tools
                    }.values()
                )
                state.completed = all(game.done for game in games)
                return state
            await prepare_sample([*main_tasks, *side_tasks])
            # A replay is offered both editors, so a recording made with either dispatches.
            tools = build_sample_tools(
                state,
                main_tasks,
                side_tasks,
                text_editor_config="both"
                if white.split_editor or active_replay_sandbox() is not None
                else "default",
            )
            policy: Policy = playing_info.function(
                **(dict(playing_args) if playing_args else {})
            )
            # A task-test tool is absent on a sample without that task, so one config covers both.
            if (
                missing := policy.private_tools
                - set(tools.private_tools)
                - TASK_TEST_TOOLS
            ):
                raise ValidationError(
                    f"policy {playing_info.name!r} uses private tool(s) {sorted(missing)} that this "
                    "sample's environment does not declare"
                )
            # Built only from what white allows, so a policy that ignores its arg still gets no more.
            chosen = {
                name: tool
                for name, tool in sorted(tools.private_tools.items())
                if name in policy.private_tools and name in allowed_tools
            }
            policy.setup(RedView(meta.environment, meta.main_tasks, meta.side_tasks))
            # The controller holds the capabilities the policy must not touch, and drives its loop.
            untrusted = Untrusted(policy, side_setters=tools.side_setters)
            untrusted.private_tools = {**tools.side_setters, **chosen}

            protocol: Protocol = protocol_info.function(**protocol_kwargs)
            game = Game(
                state=state,
                untrusted=untrusted,
                white=white,
                protocol=protocol,
                tools=tools,
            )
            return await game.run()

        return solve

    return eval_runner()

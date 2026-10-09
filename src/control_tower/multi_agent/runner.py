"""Run independent Inspect agents in one sample and retain their transcripts."""

import asyncio
from dataclasses import dataclass

from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessage
from inspect_ai.util import StoreModel, span, store_as
from pydantic import Field


@dataclass
class AgentRun:
    """An agent and the private conversation it starts with."""

    agent_id: str
    agent: Agent
    messages: list[ChatMessage]


class MultiAgentTranscripts(StoreModel):
    """Per-agent conversations, including partial work from failed runs."""

    by_agent: dict[str, list[ChatMessage]] = Field(default_factory=dict)
    outcomes: dict[str, str] = Field(default_factory=dict)


async def run_agents(
    runs: list[AgentRun], *, parallel: bool = True
) -> dict[str, AgentState]:
    """Execute a fixed roster; failure cancels and awaits concurrent siblings."""
    ids = [run.agent_id for run in runs]
    if not ids or any(not name for name in ids) or len(set(ids)) != len(ids):
        raise ValueError("Agents require nonempty, unique IDs")
    transcripts = store_as(MultiAgentTranscripts)
    states = {
        run.agent_id: AgentState(
            messages=[m.model_copy(deep=True) for m in run.messages]
        )
        for run in runs
    }

    for run in runs:
        transcripts.by_agent[run.agent_id] = list(states[run.agent_id].messages)
        transcripts.outcomes[run.agent_id] = "not_started"

    async def execute(run: AgentRun) -> None:
        state = states[run.agent_id]
        outcome = "failed"
        try:
            async with span(run.agent_id, type="agent"):
                state = await run.agent(state)
            states[run.agent_id] = state
            outcome = "completed"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            transcripts.by_agent[run.agent_id] = list(state.messages)
            transcripts.outcomes[run.agent_id] = outcome

    if parallel:
        async with asyncio.TaskGroup() as group:
            for run in runs:
                group.create_task(execute(run), name=run.agent_id)
    else:
        for run in runs:
            await execute(run)
    incomplete = [name for name in ids if transcripts.outcomes[name] != "completed"]
    if incomplete:
        raise RuntimeError(f"Agents did not complete: {', '.join(incomplete)}")
    return states

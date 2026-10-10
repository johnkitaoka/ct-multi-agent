"""Public, post-run monitor views of a roster's seated actions."""

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from inspect_ai.event import Event, ModelEvent, ToolEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from pydantic import BaseModel

from control_tower.eval_logs.seat import roster_events

MESSAGE_TOOLS = frozenset({"send_message", "read_messages", "post_message"})
HIDDEN_TAGS = (
    "covert-goal",
    "private-condition",
    "red-team-guidance",
    "communication-discipline",
    "coercive-guidance",
    "monitor-subversion",
    "statefulness-experience",
    "distributed-collusion",
    "distributed-review-workflow",
    "reporting-guidance",
)
HIDDEN_BRACKETS = ("Covert Goal", "Private Condition", "Red Team Guidance")


def public_text(value: str) -> str:
    """Remove hidden instruction blocks, including an unterminated opening tag."""
    for tag in HIDDEN_TAGS:
        value = re.sub(
            rf"\s*<{tag}\b[^>]*>.*?</{tag}\s*>",
            "",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )
    for label in HIDDEN_BRACKETS:
        value = re.sub(
            rf"\s*\[{label}\].*?\[/{label}\]",
            "",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )
    openings = [
        match.start()
        for tag in HIDDEN_TAGS
        for match in re.finditer(rf"<{tag}\b[^>]*>", value, re.IGNORECASE)
    ]
    openings.extend(
        match.start()
        for label in HIDDEN_BRACKETS
        for match in re.finditer(rf"\[{label}\]", value, re.IGNORECASE)
    )
    return (value[: min(openings)] if openings else value).rstrip()


def _public_value(value: Any) -> Any:
    if isinstance(value, str):
        return public_text(value)
    if isinstance(value, ContentText):
        return value.model_copy(update={"text": public_text(value.text)})
    if isinstance(value, ContentReasoning):
        return None
    if isinstance(value, dict):
        return {key: _public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean for item in value if (clean := _public_value(item)) is not None]
    return value


def _action(message: ChatMessageAssistant) -> ChatMessageAssistant | None:
    calls = [
        replace(call, arguments=_public_value(call.arguments))
        for call in message.tool_calls or []
        if call.function not in MESSAGE_TOOLS
    ]
    text = public_text(message.text)
    if not text and not calls:
        return None
    return message.model_copy(
        update={
            "content": [ContentText(text=text)] if text else [],
            "tool_calls": calls or None,
            "metadata": None,
            "source": None,
            "model": None,
        }
    )


def _result(event: ToolEvent) -> ChatMessageTool:
    content = _public_value(event.result)
    if isinstance(content, BaseModel):
        content = [content]
    elif not isinstance(content, str | list):
        content = str(content)
    return ChatMessageTool(
        content=content,
        tool_call_id=event.id,
        function=event.function,
        error=event.error,
    )


def team_trajectories(
    events: Sequence[Event], agent_ids: list[str]
) -> tuple[dict[str, list[ChatMessage]], list[ChatMessage]]:
    """Per-worker and timestamped, worker-grouped joint views of public seated work."""
    per_agent: dict[str, list[ChatMessage]] = {agent_id: [] for agent_id in agent_ids}
    stamped: dict[str, list[tuple[Event, ChatMessage]]] = {
        agent_id: [] for agent_id in agent_ids
    }
    allowed: set[str] = set()
    for agent_id, event in roster_events(events, agent_ids):
        message: ChatMessage | None = None
        if isinstance(event, ModelEvent):
            message = _action(event.output.message)
            if isinstance(message, ChatMessageAssistant):
                allowed.update(call.id for call in message.tool_calls or [])
        elif isinstance(event, ToolEvent) and event.id in allowed:
            message = _result(event)
        if message is None:
            continue
        per_agent[agent_id].append(message)
        stamped[agent_id].append((event, message))
    joint: list[ChatMessage] = []
    for agent_id in agent_ids:
        joint.append(ChatMessageUser(content=f"===== Worker {agent_id} ====="))
        for event, message in stamped[agent_id]:
            stamp = event.timestamp.isoformat(timespec="milliseconds")
            joint.extend([ChatMessageUser(content=f"At {stamp}"), message])
    return per_agent, joint

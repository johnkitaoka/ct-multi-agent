"""Agent configuration, role tools, and an optional sample-scoped mailbox."""

import json
from collections.abc import Callable

from inspect_ai.tool import Tool, ToolDef
from inspect_ai.util import StoreModel, store_as
from pydantic import BaseModel, Field

from control_tower.utils.errors import ValidationError
from control_tower.utils.function_registry.base_model import BaseModelPython


class AgentConfig(BaseModelPython):
    """Assign a task and tools to one agent; no task ID means all assigned mains."""

    agent_id: str = Field(min_length=1)
    role: str = ""
    max_steps: int | None = Field(default=None, ge=1)
    main_task_id: str | None = None
    user: str | None = None
    tool_group: str | None = None
    instructions: str = ""
    side_instructions: str = ""
    side_task_aware: bool = False


class MailboxMessage(BaseModel):
    sender: str
    recipient: str
    content: str
    topic: str = "general"


class MailboxState(StoreModel):
    messages: list[MailboxMessage] = Field(default_factory=list)


def mailbox_tools(agent_id: str, recipients: list[str]) -> list[Tool]:
    """Expose messages addressed to this agent in the current sample."""

    async def send_message(to_agent: str, content: str, topic: str = "general") -> str:
        """Send a message to a named peer.

        Args:
            to_agent: Recipient ID from the configured roster.
            content: Message text.
            topic: Optional subject.
        """
        if to_agent not in recipients:
            raise ValueError(f"Unknown recipient: {to_agent}")
        mailbox = store_as(MailboxState)
        mailbox.messages.append(
            MailboxMessage(
                sender=agent_id, recipient=to_agent, content=content, topic=topic
            )
        )
        return f"Message sent to {to_agent}."

    async def read_messages() -> str:
        mailbox = store_as(MailboxState)
        messages = [m.model_dump() for m in mailbox.messages if m.recipient == agent_id]
        return json.dumps(messages)

    return [
        ToolDef(
            send_message,
            name="send_message",
            description="Send a message to another agent.",
        ).as_tool(),
        ToolDef(
            read_messages,
            name="read_messages",
            description="Read messages addressed to you.",
        ).as_tool(),
    ]


_TOOL_GROUPS: dict[str, Callable[[], list[Tool]]] = {}


def register_tool_group(name: str, builder: Callable[[], list[Tool]]) -> None:
    """Register the tools assigned to a named role."""
    _TOOL_GROUPS[name] = builder


def build_tool_group(name: str) -> list[Tool]:
    """Build a fresh tool list, rejecting unknown role names."""
    if name not in _TOOL_GROUPS:
        raise ValidationError(f"Tool group {name!r} is not registered")
    return _TOOL_GROUPS[name]()

"""Reading a run's seat off a finished log.

The engine writes events; this reads them back. Blue's seat is the one place every untrusted action
passes through, so its `untrusted` model events are what U did, and the `ToolEvent` carrying a
call's id is what that call did. Both post-hoc monitoring and the diagnostics read the seat through
here rather than each deriving the filter.
"""

from collections.abc import Iterable
from typing import cast

from inspect_ai.event import Event, ModelEvent, SpanBeginEvent, ToolEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    Content,
)

from control_tower.models.roles import UNTRUSTED_ROLE
from control_tower.protocols.eval import SEAT_PROVIDER

EVAL_TASK = "control_tower/eval"
"""The Inspect task `ct run eval` runs: locally, on the fleet, or on hawk."""

CONTROL_EVAL_TASK = "control_tower/control_eval"
"""The Inspect task `ct run control-eval` runs: an honest and an attack policy under one protocol, in one log."""


def seat_turns(events: Iterable[Event]) -> list[ModelEvent]:
    """Every action blue decided, in order. Blue's view by construction: red's drafts are
    `hidden_untrusted`, its planning `red_planner`, and its side-task submission calls private, so
    nothing of red's is filtered out here — it was never a seat turn."""
    return [
        event
        for event in events
        if isinstance(event, ModelEvent)
        and event.role == UNTRUSTED_ROLE
        and event.model.startswith(f"{SEAT_PROVIDER}/")
        and event.output.choices
    ]


def roster_events(
    events: Iterable[Event], agent_ids: Iterable[str]
) -> list[tuple[str, ModelEvent | ToolEvent]]:
    """The roster's seated actions and results in their recorded global order."""
    events = list(events)
    names = set(agent_ids)
    spans = {event.id: event for event in events if isinstance(event, SpanBeginEvent)}
    turns = seat_turns(events)
    seated = {id(event) for event in turns}

    def owner(span_id: str | None) -> str | None:
        while span_id is not None and (span := spans.get(span_id)) is not None:
            if span.type == "agent" and span.name in names:
                return span.name
            span_id = span.parent_id
        return None

    calls = {
        call.id: (owner(event.span_id), call.function)
        for event in turns
        for call in event.output.message.tool_calls or []
    }

    ordered: list[tuple[str, ModelEvent | ToolEvent]] = []
    for event in events:
        if not (
            isinstance(event, ModelEvent)
            and id(event) in seated
            or isinstance(event, ToolEvent)
            and event.id in calls
        ):
            continue
        agent_id = owner(event.span_id)
        if isinstance(event, ToolEvent) and calls[event.id] != (
            agent_id,
            event.function,
        ):
            continue
        if agent_id is not None:
            ordered.append((agent_id, event))
        elif id(event) in seated:
            raise ValueError("a roster seat turn has no worker span")
    return ordered


def _ran(events: "list[Event]") -> dict[str, ToolEvent]:
    return {event.id: event for event in events if isinstance(event, ToolEvent)}


def _answers(
    turn: ChatMessageAssistant, ran: dict[str, ToolEvent]
) -> list[ChatMessage]:
    # A call with no `ToolEvent` did not run — a submit's siblings, bar submission-tool calls, are dropped.
    return [
        ChatMessageTool(
            content=cast("str | list[Content]", done.result),
            tool_call_id=call.id,
            function=done.function,
            error=done.error,
        )
        for call in turn.tool_calls or []
        if (done := ran.get(call.id)) is not None
    ]


def untrusted_conversation(events: Iterable[Event]) -> list[ChatMessage]:
    """U's turns off the seat's events, each followed by what its calls did."""
    events = list(events)
    ran = _ran(events)
    return [
        m
        for event in seat_turns(events)
        for m in (event.output.message, *_answers(event.output.message, ran))
    ]


def action_views(events: Iterable[Event]) -> list[tuple[ModelEvent, list[ChatMessage]]]:
    """Each action, with what U had done before it: the conversation a monitor judges it against."""
    events = list(events)
    ran = _ran(events)
    views, conversation = [], []
    for event in seat_turns(events):
        turn = event.output.message
        if turn.tool_calls:
            views.append((event, [*conversation, turn]))
        conversation.extend((turn, *_answers(turn, ran)))
    return views

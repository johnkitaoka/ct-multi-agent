"""Shared helpers for parsing and rendering Inspect transcript logs."""

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class Span:
    id: str
    name: str
    parent_id: str | None
    start: Any
    end: Any = None
    type: str | None = None


@dataclass
class BackgroundAgent:
    handle: str
    span_id: str
    name: str
    prompt: str
    task_description: str
    dispatched: Any
    dispatch_completed: Any
    background: bool
    thread_type: str = "subagent"


@dataclass
class TaskContext:
    sample_id: Any
    sample_input: str
    target: str
    metadata: dict[str, Any]
    sandbox: str
    system_messages: list[str]
    tools: list[dict[str, Any]]


@dataclass(frozen=True)
class Trace:
    """Normalized lane assignment and context for one Inspect sample."""

    context: TaskContext
    events: tuple[Any, ...]
    spans: dict[str, Span]
    agents_by_span: dict[str, BackgroundAgent]
    events_by_lane: dict[str, tuple[Any, ...]]
    lane_order: tuple[str, ...]
    tools_by_lane: dict[str, tuple[dict[str, Any], ...]]
    start: Any
    end: Any


def reltime(ts: Any, t0: Any) -> str:
    if ts is None or t0 is None:
        return "+?:??.??"
    seconds = max(0.0, (ts - t0).total_seconds())
    return f"+{int(seconds // 60)}:{seconds % 60:05.2f}"


def duration(start: Any, end: Any) -> str:
    if start is None or end is None:
        return "?"
    return f"{(end - start).total_seconds():.2f}s"


def trim(text: str, limit: int | None) -> str:
    text = text.strip()
    if limit is None:
        return text
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[+{len(text) - limit} chars]"


def indented(text: str, spaces: int = 4) -> list[str]:
    prefix = " " * spaces
    if not text.strip():
        return [prefix + "(empty)"]
    return [prefix + line for line in text.strip().splitlines()]


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def message_content_parts(message: Any) -> tuple[list[str], list[str]]:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return [], [content.strip()] if content.strip() else []
    if not isinstance(content, list):
        text = str(content).strip()
        return [], [text] if text else []

    reasoning: list[str] = []
    text: list[str] = []
    for part in content:
        part_type = getattr(part, "type", "")
        if part_type == "reasoning":
            value = getattr(part, "reasoning", None) or getattr(part, "summary", None)
            if value:
                reasoning.append(str(value).strip())
        else:
            value = getattr(part, "text", None)
            if value:
                text.append(str(value).strip())
    return [p for p in reasoning if p], [p for p in text if p]


def to_plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(v) for v in value]
    return value


def json_text(value: Any, limit: int | None) -> str:
    try:
        text = json.dumps(to_plain_data(value), indent=2, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return trim(text, limit)


def first_system_messages(sample: Any) -> list[str]:
    messages = getattr(sample, "messages", None) or []
    out: list[str] = []
    for message in messages:
        if getattr(message, "role", "") == "system":
            text = message_text(message)
            if text:
                out.append(text)
    return out


def first_event_tools(events: Iterable[Any]) -> list[dict[str, Any]]:
    for event in events:
        if getattr(event, "event", None) != "model":
            continue
        tools = getattr(event, "tools", None) or []
        return [to_plain_data(tool) for tool in tools]
    return []


def collect_task_context(sample: Any, events: Iterable[Any]) -> TaskContext:
    return TaskContext(
        sample_id=getattr(sample, "id", ""),
        sample_input=str(getattr(sample, "input", "") or ""),
        target=str(getattr(sample, "target", "") or ""),
        metadata=dict(getattr(sample, "metadata", {}) or {}),
        sandbox=str(getattr(sample, "sandbox", "") or ""),
        system_messages=first_system_messages(sample),
        tools=first_event_tools(events),
    )


def event_time_bounds(events: Iterable[Any]) -> tuple[Any, Any]:
    timestamps = [getattr(event, "timestamp", None) for event in events]
    timestamps = [ts for ts in timestamps if ts is not None]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def collect_spans(events: Iterable[Any]) -> dict[str, Span]:
    spans: dict[str, Span] = {}
    for event in events:
        if event.event == "span_begin":
            spans[event.id] = Span(
                id=event.id,
                name=getattr(event, "name", "") or "",
                parent_id=getattr(event, "parent_id", None),
                start=getattr(event, "timestamp", None),
                type=getattr(event, "type", None),
            )
        elif event.event == "span_end" and event.id in spans:
            spans[event.id].end = getattr(event, "timestamp", None)
    return spans


def ancestor_span_ids(span_id: str | None, spans: dict[str, Span]) -> list[str]:
    ids: list[str] = []
    current = span_id
    for _ in range(100):
        if current is None or current not in spans:
            break
        ids.append(current)
        current = spans[current].parent_id
    return ids


def parse_handle(result: str) -> str | None:
    match = re.search(r"\bAGENT-\d+\b", result)
    return match.group(0) if match else None


def collect_agent_threads(
    events: Iterable[Any],
    spans: dict[str, Span],
) -> dict[str, BackgroundAgent]:
    events = list(events)
    agents: dict[str, BackgroundAgent] = {}
    sync_count = 0
    for event in events:
        if event.event != "tool" or getattr(event, "function", None) != "agent":
            continue
        span_id = getattr(event, "agent_span_id", None)
        if span_id is None:
            continue
        args = getattr(event, "arguments", {}) or {}
        handle = parse_handle(str(getattr(event, "result", "")))
        is_background = bool(args.get("background")) or handle is not None
        if handle is None:
            sync_count += 1
            handle = f"SUBAGENT-{sync_count}"
        fallback = spans.get(span_id, Span(span_id, "", None, None)).name
        agents[span_id] = BackgroundAgent(
            handle=handle,
            span_id=span_id,
            name=str(args.get("subagent_type") or fallback),
            prompt=str(args.get("prompt") or ""),
            task_description=str(args.get("task_description") or ""),
            dispatched=getattr(event, "timestamp", None),
            dispatch_completed=getattr(event, "completed", None),
            background=is_background,
        )

    for span in spans.values():
        if span.type != "agent" or span.id in agents:
            continue
        # Nested react spans belong to their enclosing roster agent.
        if any(parent in agents for parent in ancestor_span_ids(span.parent_id, spans)):
            continue
        handle = span.name or span.id
        if any(agent.handle == handle for agent in agents.values()):
            handle = f"{handle} ({span.id})"
        agents[span.id] = BackgroundAgent(
            handle=handle,
            span_id=span.id,
            name=span.name or handle,
            prompt="",
            task_description="",
            dispatched=span.start,
            dispatch_completed=span.end,
            background=False,
            thread_type="peer",
        )

    model_spans_by_parent: dict[str, list[str]] = defaultdict(list)
    for event in events:
        if event.event != "model":
            continue
        span_id = getattr(event, "span_id", None)
        if span_id is None:
            continue
        span = spans.get(span_id)
        if (
            span is None
            or span.parent_id is None
            or any(ancestor in agents for ancestor in ancestor_span_ids(span_id, spans))
        ):
            continue
        if span_id not in model_spans_by_parent[span.parent_id]:
            model_spans_by_parent[span.parent_id].append(span_id)

    peer_count = 0
    for span_ids in model_spans_by_parent.values():
        if len(span_ids) < 2:
            continue
        for span_id in span_ids:
            if span_id in agents:
                continue
            span = spans[span_id]
            peer_count += 1
            handle = span.name or f"PEER-{peer_count}"
            agents[span_id] = BackgroundAgent(
                handle=handle,
                span_id=span_id,
                name=span.name or handle,
                prompt="",
                task_description="",
                dispatched=span.start,
                dispatch_completed=span.end,
                background=False,
                thread_type="peer",
            )
    return agents


def thread_key(
    event: Any,
    spans: dict[str, Span],
    agents: dict[str, BackgroundAgent],
) -> str:
    for span_id in ancestor_span_ids(getattr(event, "span_id", None), spans):
        if span_id in agents:
            return agents[span_id].handle
    return "ORCHESTRATOR"


def parse_trace(sample: Any) -> Trace:
    """Parse raw Inspect events once for all transcript renderers."""

    events = tuple(getattr(sample, "events", None) or ())
    spans = collect_spans(events)
    agents_by_span = collect_agent_threads(events, spans)
    context = collect_task_context(sample, events)
    start, end = event_time_bounds(events)

    events_by_lane_lists: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        if getattr(event, "event", None) in {"model", "tool", "info"}:
            lane = thread_key(event, spans, agents_by_span)
            events_by_lane_lists[lane].append(event)

    agent_ids = context.metadata.get("agent_ids") or []
    if (
        not agents_by_span
        and len(agent_ids) == 1
        and set(events_by_lane_lists) == {"ORCHESTRATOR"}
    ):
        events_by_lane_lists[str(agent_ids[0])] = events_by_lane_lists.pop(
            "ORCHESTRATOR"
        )

    agent_handles = {agent.handle for agent in agents_by_span.values()}
    lane_order: list[str] = []
    if events_by_lane_lists.get("ORCHESTRATOR"):
        lane_order.append("ORCHESTRATOR")
    lane_order.extend(
        sorted(
            agent_handles
            | {lane for lane in events_by_lane_lists if lane != "ORCHESTRATOR"}
        )
    )

    tools_by_lane: dict[str, tuple[dict[str, Any], ...]] = {}
    for lane, lane_events in events_by_lane_lists.items():
        for event in lane_events:
            if getattr(event, "event", None) != "model":
                continue
            tools = getattr(event, "tools", None) or []
            tools_by_lane[lane] = tuple(to_plain_data(tool) for tool in tools)
            break

    return Trace(
        context=context,
        events=events,
        spans=spans,
        agents_by_span=agents_by_span,
        events_by_lane={
            lane: tuple(lane_events)
            for lane, lane_events in events_by_lane_lists.items()
        },
        lane_order=tuple(lane_order),
        tools_by_lane=tools_by_lane,
        start=start,
        end=end,
    )


def input_message_key(message: Any) -> tuple[str, str]:
    return str(getattr(message, "role", "") or "message"), message_text(message)


def input_message_label(role: str, text: str) -> str:
    if role == "system":
        return "system prompt"
    if role == "developer":
        return "developer instruction"
    if is_incoming_message(role, text):
        return "message"
    if is_notification_message(role, text):
        return "notification"
    if role == "user" and text.startswith("[Automatic"):
        return "user/system update"
    if role == "user":
        return "user"
    return f"{role} input"


def is_notification_message(role: str, text: str) -> bool:
    return role == "user" and (
        text.startswith("[Coordination update]")
        or text.startswith("[Incoming message #")
    )


def is_incoming_message(role: str, text: str) -> bool:
    return role == "user" and text.startswith("[Incoming message #")


def should_render_input_message(role: str, text: str) -> bool:
    if role in {"system", "developer", "user"}:
        return True
    return False


def should_dedupe_input_message(role: str, text: str) -> bool:
    return True

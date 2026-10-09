#!/usr/bin/env python3
"""Render Inspect eval logs as interactive HTML execution transcripts."""

import argparse
import glob
import hashlib
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from pygments import highlight as pygments_highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.lexers.special import TextLexer
from pygments.util import ClassNotFound

from control_tower.eval_logs.html.parse import (
    BackgroundAgent,
    Span,
    TaskContext,
    duration,
    input_message_key,
    input_message_label,
    is_notification_message,
    json_text,
    message_content_parts,
    parse_handle,
    parse_trace,
    reltime,
    should_dedupe_input_message,
    should_render_input_message,
    trim,
)

LIFECYCLE_TOOLS = {"agent_status", "agent_wait", "agent_cancel", "agent_list"}
SUBMIT_TOOLS = {"submit", "submit_answer", "end_task"}
MESSAGE_SEND_TOOLS = {"send_message", "post_message"}


@dataclass
class HtmlBlock:
    lane: str
    timestamp: Any
    kind: str
    title: str
    body: str
    reltime: str
    render_as: str = "markdown"
    language: str | None = None
    tool_name: str | None = None


def iter_eval_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        files.extend(matches if matches else [pattern])
    return sorted({path for path in files if path.endswith(".eval")})


def safe_filename_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def count_message_sends(events: list[Any]) -> int:
    return sum(
        1
        for event in events
        if getattr(event, "event", None) == "tool"
        and getattr(event, "function", None) in MESSAGE_SEND_TOOLS
    )


def estimate_tokens(text: str) -> int:
    return round(len(text) / 4)


def format_tokens_k(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k"


def full_hidden_reasoning_token_estimate(
    context: TaskContext,
    blocks_by_lane: dict[str, list[HtmlBlock]],
) -> int:
    """Estimate the untrimmed transcript content rendered by the HTML viewer."""

    parts = [
        context.sample_input,
        context.target,
        json.dumps(context.metadata, sort_keys=True, default=str),
        *context.system_messages,
    ]
    for blocks in blocks_by_lane.values():
        for block in blocks:
            if block.tool_name:
                parts.append(block.tool_name)
            parts.append(block.body)
    return estimate_tokens("\n".join(parts))


def html_output_path(out_dir: Path, eval_file: str, sample_id: Any) -> Path:
    safe_stem = safe_filename_part(Path(eval_file).stem)
    safe_id = safe_filename_part(sample_id)
    name = f"{safe_stem}__sample_{safe_id}"
    if len(name) > 220:
        digest = hashlib.sha256(name.encode()).hexdigest()[:12]
        name = f"{name[:206]}__{digest}"
    return out_dir / f"{name}.html"


def inline_markdown_html(text: str) -> str:
    code_spans: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_spans.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"__CODE_SPAN_{len(code_spans) - 1}__"

    text = re.sub(r"`([^`]+)`", stash_code, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(
        r"(?<!\*)\*([^*\s][^*\n]*?[^*\s])\*(?!\*)",
        r"<em>\1</em>",
        text,
    )
    for idx, code in enumerate(code_spans):
        text = text.replace(f"__CODE_SPAN_{idx}__", code)
    return text


def render_code_html(text: str, language: str | None, limit: int | None) -> str:
    source = trim(text, limit)
    lang = language or "text"
    try:
        lexer = get_lexer_by_name(lang)
    except ClassNotFound:
        lexer = TextLexer()
    highlighted = pygments_highlight(source, lexer, HtmlFormatter(nowrap=True)).rstrip()
    safe_lang = re.sub(r"[^a-zA-Z0-9_-]", "", lang)
    return f'<pre class="code language-{safe_lang}"><code>{highlighted}</code></pre>'


def render_markdown_html(text: str, limit: int | None) -> str:
    lines = trim(text, limit).splitlines()
    rendered: list[str] = []
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        # Reflow SOURCE-wrapped prose instead of forcing a <br> per line. Prompt text is wrapped in
        # the Python source at ~100 chars, and a <br> on every line made each source line its own
        # display line -- ragged, and never matching the column width. A line is treated as the
        # continuation of a wrapped sentence when the line before it is long and does not end in a
        # markdown hard break (two trailing spaces); deliberate short lines like "==== ROLE ====",
        # numbered steps and labels keep their break because the line before them is short.
        if paragraph:
            parts: list[str] = []
            for line in paragraph:
                prev = parts[-1] if parts else None
                soft_wrap = (
                    prev is not None
                    and len(prev.rstrip()) >= 75
                    and not prev.endswith("  ")
                )
                if soft_wrap:
                    parts[-1] = f"{parts[-1].rstrip()} {line.lstrip()}"
                else:
                    parts.append(line)
            content = "<br>".join(inline_markdown_html(p) for p in parts)
            rendered.append(f"<p>{content}</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "":
            flush_paragraph()
            i += 1
            continue

        fence = re.match(r"^```([a-zA-Z0-9_-]+)?\s*$", stripped)
        if fence:
            flush_paragraph()
            lang = fence.group(1) or "text"
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            rendered.append(render_code_html("\n".join(code_lines), lang, limit))
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = min(len(heading.group(1)) + 2, 6)
            rendered.append(
                f"<h{level}>{inline_markdown_html(heading.group(2))}</h{level}>"
            )
            i += 1
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet:
            flush_paragraph()
            items: list[str] = []
            while i < len(lines):
                match = re.match(r"^[-*]\s+(.+)$", lines[i].strip())
                if not match:
                    break
                items.append(f"<li>{inline_markdown_html(match.group(1))}</li>")
                i += 1
            rendered.append("<ul>" + "".join(items) + "</ul>")
            continue

        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if ordered:
            flush_paragraph()
            items = []
            while i < len(lines):
                match = re.match(r"^\d+[.)]\s+(.+)$", lines[i].strip())
                if not match:
                    break
                items.append(f"<li>{inline_markdown_html(match.group(1))}</li>")
                i += 1
            rendered.append("<ol>" + "".join(items) + "</ol>")
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    if not rendered:
        return "<p>(empty)</p>"
    return "\n".join(rendered)


def looks_like_python(text: str) -> bool:
    python_markers = (
        r"^\s*(def|class|import|from|return|assert|with|if|elif|else|for|while)\b",
        r"^\s*@\w+",
        r"^\s*[a-zA-Z_]\w*\s*=",
    )
    return any(re.search(pattern, text, re.MULTILINE) for pattern in python_markers)


def detect_result_language(function: str, text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "text"
    if function == "python":
        return "python"
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            return "text"
    if function == "bash":
        shell_output_markers = (
            "test session starts",
            "FAILURES",
            "FAILED",
            "PASSED",
            "AssertionError",
            "Traceback",
            "Requirement already satisfied",
            "Defaulting to user installation",
        )
        if any(marker in stripped for marker in shell_output_markers):
            return "bash"
        if looks_like_python(stripped):
            return "python"
        return "bash"
    if looks_like_python(stripped):
        return "python"
    return "text"


def pygments_css() -> str:
    if HtmlFormatter is None:
        return ""
    return HtmlFormatter(style="friendly").get_style_defs(".code")


def render_block_body(block: HtmlBlock, result_trim: int | None) -> str:
    if block.render_as == "tool-call":
        tool = html.escape(block.tool_name or "tool")
        try:
            args = json.loads(block.body)
        except Exception:
            args = {}
        command = args.get("command") if isinstance(args, dict) else None
        if isinstance(command, str) and block.tool_name in {"bash", "python"}:
            lang = "bash" if block.tool_name == "bash" else "python"
            return (
                f'<div class="tool-name">{tool}</div>'
                '<div class="arg-label">command</div>'
                + render_code_html(command, lang, result_trim)
            )
        return f'<div class="tool-name">{tool}</div>' + render_code_html(
            block.body, block.language or "json", result_trim
        )
    if block.render_as == "code":
        return render_code_html(block.body, block.language, result_trim)
    return f'<div class="markdown-body">{render_markdown_html(block.body, result_trim)}</div>'


def lane_label(handle: str, agents_by_handle: dict[str, BackgroundAgent]) -> str:
    if handle == "ORCHESTRATOR":
        return "Orchestrator"
    agent = agents_by_handle.get(handle)
    if agent is None:
        return handle
    return (
        agent.handle
        if agent.handle == agent.name
        else f"{agent.handle} {agent.name}".strip()
    )


def tool_call_text(tool_call: Any) -> str:
    return json.dumps(
        getattr(tool_call, "arguments", {}) or {},
        indent=2,
        sort_keys=True,
    )


def html_input_kind(role: str, text: str) -> str:
    if role in {"system", "developer"}:
        return role
    if role == "user":
        return "notification" if is_notification_message(role, text) else "user"
    return "input-message"


def html_input_title(rel: str, role: str, text: str) -> str:
    label = input_message_label(role, text)
    if label == "user":
        return f"{rel} input"
    return f"{rel} {label}"


def add_input_html_blocks(
    blocks: list[HtmlBlock],
    event: Any,
    *,
    lane: str,
    t0: Any,
    seen_input_messages: set[tuple[str, str]],
    result_trim: int | None,
    include_user_updates: bool,
) -> None:
    ts = getattr(event, "timestamp", None)
    rel = reltime(ts, t0)
    for message in getattr(event, "input", []) or []:
        role, text = input_message_key(message)
        if not should_render_input_message(role, text):
            continue
        if role != "system" and not include_user_updates:
            continue
        if not text:
            continue
        if should_dedupe_input_message(role, text):
            if (role, text) in seen_input_messages:
                continue
            seen_input_messages.add((role, text))
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind=html_input_kind(role, text),
                title=html_input_title(rel, role, text),
                body=trim(text, result_trim),
                reltime=rel,
                render_as="markdown",
            )
        )


def add_model_html_blocks(
    blocks: list[HtmlBlock],
    event: Any,
    *,
    lane: str,
    t0: Any,
    turn: int,
    seen_input_messages: set[tuple[str, str]],
    result_trim: int | None,
    include_user_updates: bool,
) -> None:
    ts = getattr(event, "timestamp", None)
    rel = reltime(ts, t0)
    add_input_html_blocks(
        blocks,
        event,
        lane=lane,
        t0=t0,
        seen_input_messages=seen_input_messages,
        result_trim=result_trim,
        include_user_updates=include_user_updates,
    )

    output = getattr(event, "output", None)
    if output is None or output.empty:
        return

    message = output.message
    reasoning_parts, text_parts = message_content_parts(message)
    for part in reasoning_parts:
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind="reasoning",
                title=f"{rel} turn {turn} reasoning",
                body=trim(part, result_trim),
                reltime=rel,
                render_as="markdown",
            )
        )

    text = "\n\n".join(text_parts).strip()
    if text:
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind="message",
                title=f"{rel} turn {turn} agent message",
                body=trim(text, result_trim),
                reltime=rel,
                render_as="markdown",
            )
        )

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", "")
        kind = "tool-call"
        title = f"{rel} turn {turn} tool call"
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind=kind,
                title=title,
                body=trim(tool_call_text(tool_call), result_trim),
                reltime=rel,
                render_as="tool-call",
                language="json",
                tool_name=function,
            )
        )

    if not text and not tool_calls and not reasoning_parts:
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind="message",
                title=f"{rel} turn {turn} agent message",
                body="(empty output)",
                reltime=rel,
                render_as="markdown",
            )
        )


def add_tool_html_block(
    blocks: list[HtmlBlock],
    event: Any,
    *,
    lane: str,
    t0: Any,
    result_trim: int | None,
) -> None:
    ts = getattr(event, "completed", None) or getattr(event, "timestamp", None)
    rel = reltime(ts, t0)
    function = getattr(event, "function", "")
    label = "lifecycle result" if function in LIFECYCLE_TOOLS else "tool result"
    result = getattr(event, "result", "")
    error = getattr(event, "error", None)
    if error is not None:
        result = getattr(error, "message", str(error))
    handle = parse_handle(str(result))
    suffix = f" -> {handle}" if function == "agent" and handle else ""
    text = str(result)
    is_submit = function in SUBMIT_TOOLS
    blocks.append(
        HtmlBlock(
            lane=lane,
            timestamp=ts,
            kind="tool-result",
            title=f"{rel} {label}: {function}{suffix}",
            body=trim(text, result_trim),
            reltime=rel,
            render_as="markdown" if is_submit else "code",
            language=None if is_submit else detect_result_language(function, text),
        )
    )


def add_info_html_block(
    blocks: list[HtmlBlock],
    event: Any,
    *,
    lane: str,
    t0: Any,
    result_trim: int | None,
) -> None:
    ts = getattr(event, "timestamp", None)
    rel = reltime(ts, t0)
    data = getattr(event, "data", "")
    text = data if isinstance(data, str) else json.dumps(data, sort_keys=True)
    blocks.append(
        HtmlBlock(
            lane=lane,
            timestamp=ts,
            kind="info",
            title=f"{rel} info",
            body=trim(text, result_trim),
            reltime=rel,
            render_as="markdown",
        )
    )


def collect_html_blocks(
    by_thread: dict[str, list[Any]],
    t0: Any,
    result_trim: int | None = None,
) -> dict[str, list[HtmlBlock]]:
    blocks_by_lane: dict[str, list[HtmlBlock]] = {}
    for lane, lane_events in by_thread.items():
        blocks: list[HtmlBlock] = []
        seen_input_messages: set[tuple[str, str]] = set()
        turn = 1
        model_tool_ids = {
            call.id
            for event in lane_events
            if event.event == "model"
            for choice in (
                getattr(getattr(event, "output", None), "choices", None) or []
            )
            for call in (getattr(choice.message, "tool_calls", None) or [])
        }
        for event in lane_events:
            if getattr(event, "hidden_from_timeline", False):
                continue
            if event.event == "model":
                add_model_html_blocks(
                    blocks,
                    event,
                    lane=lane,
                    t0=t0,
                    turn=turn,
                    seen_input_messages=seen_input_messages,
                    result_trim=result_trim,
                    include_user_updates=True,
                )
                turn += 1
            elif event.event == "tool":
                if event.id not in model_tool_ids:
                    ts = getattr(event, "timestamp", None)
                    rel = reltime(ts, t0)
                    blocks.append(
                        HtmlBlock(
                            lane=lane,
                            timestamp=ts,
                            kind="tool-call",
                            title=f"{rel} tool call: {event.function}",
                            body=json_text(event.arguments, result_trim),
                            reltime=rel,
                            render_as="code",
                            language="json",
                        )
                    )
                add_tool_html_block(
                    blocks,
                    event,
                    lane=lane,
                    t0=t0,
                    result_trim=result_trim,
                )
            elif event.event == "info":
                add_info_html_block(
                    blocks,
                    event,
                    lane=lane,
                    t0=t0,
                    result_trim=result_trim,
                )
        blocks_by_lane[lane] = blocks
    return blocks_by_lane


def render_html_lane_header(
    lane: str,
    agents_by_handle: dict[str, BackgroundAgent],
    spans: dict[str, Span],
    t0: Any,
    root_label: str = "Orchestrator",
    root_meta: str = "primary thread",
) -> str:
    title = html.escape(
        root_label if lane == "ORCHESTRATOR" else lane_label(lane, agents_by_handle)
    )
    if lane == "ORCHESTRATOR":
        return f'<h2>{title}</h2><p class="lane-meta">{html.escape(root_meta)}</p>'

    agent = agents_by_handle.get(lane)
    if agent is None:
        return f'<h2>{title}</h2><p class="lane-meta">subagent thread</p>'

    span = spans.get(agent.span_id)
    active = ""
    if span is not None:
        active = f"{reltime(span.start, t0)} to {reltime(span.end, t0)}"
    if agent.thread_type == "peer":
        mode = "peer agent"
    else:
        mode = "background" if agent.background else "synchronous"
    return (
        f"<h2>{title}</h2>"
        f'<p class="lane-meta">{mode} · active {html.escape(active or "?")}</p>'
    )


def render_tool_summary(tools: list[dict[str, Any]], result_trim: int | None) -> str:
    if not tools:
        return "<p>No tool schema was found in model events.</p>"
    rows: list[str] = []
    for tool in tools:
        name = html.escape(str(tool.get("name", "?")))
        description = str(tool.get("description", "") or "")
        params = tool.get("parameters", {})
        rows.append(
            '<details class="tool-detail">'
            f"<summary>{name}</summary>"
            f'<div class="markdown-body">{render_markdown_html(description, result_trim)}</div>'
            f"{render_code_html(json_text(params, result_trim), 'json', result_trim)}"
            "</details>"
        )
    return "".join(rows)


def render_task_context_panel(
    context: TaskContext,
    result_trim: int | None,
    *,
    tools_by_lane: dict[str, tuple[dict[str, Any], ...]] | None = None,
) -> str:
    summary_bits = []
    if context.metadata:
        for key in (
            "task_family",
            "task_name",
            "docker_image",
            "category",
            "difficulty",
            "expert_time_estimate_min",
            "time_horizon_median_min",
            "compose_template",
            "tests_dir",
        ):
            value = context.metadata.get(key)
            if value not in (None, ""):
                summary_bits.append(
                    f'<span class="pill">{html.escape(key)}: '
                    f"{html.escape(str(value))}</span>"
                )
    tools_label = ", ".join(
        str(tool.get("name", "?")) for tool in context.tools if isinstance(tool, dict)
    )

    parts = [
        '<section class="task-context">',
        "<h2>Task Context</h2>",
        '<div class="context-summary">',
        *summary_bits,
        "</div>",
        "<details open><summary>input</summary>",
        f'<div class="markdown-body">{render_markdown_html(context.sample_input, result_trim)}</div>',
        "</details>",
    ]
    if context.target:
        parts.extend(
            [
                "<details><summary>target</summary>",
                f'<div class="markdown-body">{render_markdown_html(context.target, result_trim)}</div>',
                "</details>",
            ]
        )
    if context.metadata:
        parts.extend(
            [
                "<details><summary>metadata</summary>",
                render_code_html(
                    json_text(context.metadata, result_trim), "json", result_trim
                ),
                "</details>",
            ]
        )
    if context.sandbox:
        parts.extend(
            [
                "<details><summary>sandbox</summary>",
                render_code_html(context.sandbox, "text", result_trim),
                "</details>",
            ]
        )
    if context.system_messages:
        parts.extend(
            [
                "<details><summary>system prompt</summary>",
                f'<div class="markdown-body">{render_markdown_html(context.system_messages[0], result_trim)}</div>',
                "</details>",
            ]
        )
    if tools_by_lane:
        for lane, tools in tools_by_lane.items():
            lane_tools_label = ", ".join(
                str(tool.get("name", "?")) for tool in tools if isinstance(tool, dict)
            )
            parts.extend(
                [
                    "<details>",
                    f"<summary>{html.escape(lane)} tools: "
                    f"{html.escape(lane_tools_label)}</summary>",
                    render_tool_summary(list(tools), result_trim),
                    "</details>",
                ]
            )
    elif context.tools:
        parts.extend(
            [
                f"<details><summary>available tools: {html.escape(tools_label)}</summary>",
                render_tool_summary(context.tools, result_trim),
                "</details>",
            ]
        )
    parts.append("</section>")
    return "".join(parts)


def block_kind_order(kind: str) -> int:
    order = {
        "system": 0,
        "developer": 1,
        "user": 2,
        "notification": 3,
        "input-message": 4,
        "reasoning": 5,
        "message": 6,
        "tool-call": 7,
        "tool-result": 8,
        "info": 9,
    }
    return order.get(kind, 50)


def block_sort_key(block: HtmlBlock) -> tuple[int, Any, int, str]:
    if block.timestamp is None:
        return (1, "", block_kind_order(block.kind), block.title)
    return (0, block.timestamp, block_kind_order(block.kind), block.title)


def timestamp_sort_key(timestamp: Any) -> tuple[int, Any]:
    if timestamp is None:
        return (1, "")
    return (0, timestamp)


def synced_time_label(timestamp: Any, t0: Any) -> str:
    return reltime(timestamp, t0)


def render_html_block(block: HtmlBlock, result_trim: int | None) -> str:
    return (
        '<article class="block '
        f'{html.escape(block.kind)}">'
        f'<div class="block-title">{html.escape(block.title)}</div>'
        f'<div class="block-content">'
        f"{render_block_body(block, result_trim)}"
        "</div>"
        "</article>"
    )


def render_trace_grid(
    *,
    lane_order: list[str],
    blocks_by_lane: dict[str, list[HtmlBlock]],
    agents_by_handle: dict[str, BackgroundAgent],
    spans: dict[str, Span],
    t0: Any,
    result_trim: int | None = None,
    root_label: str = "Orchestrator",
    root_meta: str = "primary thread",
    sync_time_axis: bool = False,
) -> str:
    lane_count = max(1, len(lane_order))
    if sync_time_axis:
        bucket_min: dict[str, Any] = {}
        by_lane_time: dict[tuple[str, Any], list[HtmlBlock]] = defaultdict(list)
        for lane, blocks in blocks_by_lane.items():
            for block in sorted(blocks, key=block_sort_key):
                label = synced_time_label(block.timestamp, t0)
                by_lane_time[(lane, label)].append(block)
                current = bucket_min.get(label)
                if current is None or timestamp_sort_key(
                    block.timestamp
                ) < timestamp_sort_key(current):
                    bucket_min[label] = block.timestamp
        row_keys = sorted(
            bucket_min,
            key=lambda label: timestamp_sort_key(bucket_min[label]),
        )
        grid_class = "trace-grid synced-time-grid"
    else:
        row_keys = sorted(
            {block.timestamp for blocks in blocks_by_lane.values() for block in blocks},
            key=timestamp_sort_key,
        )
        by_lane_time = defaultdict(list)
        for lane, blocks in blocks_by_lane.items():
            for block in sorted(blocks, key=block_sort_key):
                by_lane_time[(lane, block.timestamp)].append(block)
        grid_class = "trace-grid compact-time-grid"

    cells: list[str] = [
        f'<section class="{grid_class}" style="--lane-count: {lane_count};">'
    ]
    lane_start_column = 2 if sync_time_axis else 1
    if sync_time_axis:
        cells.append(
            '<section class="time-header" style="grid-column: 1; grid-row: 1;">'
            '<h2>Time</h2><p class="lane-meta">visible timestamp</p>'
            "</section>"
        )
    for column, lane in enumerate(lane_order, start=lane_start_column):
        cells.append(
            '<section class="lane-header" '
            f'style="grid-column: {column}; grid-row: 1;">'
            + render_html_lane_header(
                lane,
                agents_by_handle,
                spans,
                t0,
                root_label=root_label,
                root_meta=root_meta,
            )
            + "</section>"
        )

    last_row = len(row_keys) + 1
    for row_offset, row_key in enumerate(row_keys, start=2):
        if sync_time_axis:
            class_names = ["time-cell"]
            if row_offset == last_row:
                class_names.append("last-row")
            cells.append(
                "<div "
                f'class="{" ".join(class_names)}" '
                f'style="grid-column: 1; grid-row: {row_offset};">'
                f"{html.escape(str(row_key))}"
                "</div>"
            )
        for column, lane in enumerate(lane_order, start=lane_start_column):
            blocks = by_lane_time.get((lane, row_key), [])
            class_names = ["thread-cell"]
            if blocks:
                class_names.append("has-block")
            if row_offset == last_row:
                class_names.append("last-row")
            rendered_blocks = "".join(
                render_html_block(block, result_trim) for block in blocks
            )
            cells.append(
                "<div "
                f'class="{" ".join(class_names)}" '
                f'style="grid-column: {column}; grid-row: {row_offset};">'
                f"{rendered_blocks}"
                "</div>"
            )
    cells.append("</section>")
    return "".join(cells)


def render_monitor_box(
    trial: dict[str, Any], monitor_meta: dict[str, Any], epoch_label: str
) -> str:
    """Display supplied scores verbatim without thresholds or rescoring."""
    views = trial.get("views", {})
    chips: list[str] = []
    details: list[str] = []
    for key, cell in views.items():
        label = html.escape(str(cell.get("label", key)))
        median = html.escape(str(cell.get("median", "unavailable")))
        chips.append(f'<span class="pill"><b>{label}</b> median {median}</span>')
        reasonings = "".join(
            f"<details><summary>Read {i}: {html.escape(str(rec.get('score', 'unavailable')))}</summary>"
            f"<pre>{html.escape(str(rec.get('reasoning', '')))}</pre></details>"
            for i, rec in enumerate(cell.get("reasonings", []), start=1)
        )
        details.append(
            f"<details><summary>{label}: supplied readings</summary>"
            f"<pre>{html.escape(str(cell.get('scores', [])))}</pre>{reasonings}</details>"
        )
    return (
        '<section class="task-context monitor-box"><h2>Supplied monitor scores</h2>'
        f"<p>{html.escape(epoch_label)} · {html.escape(str(monitor_meta.get('model', '')))}</p>"
        "<p>Raw scores; no pass/fail threshold is applied.</p>"
        f'<div class="meta">{"".join(chips)}</div>{"".join(details)}</section>'
    )


def render_saved_conversations(sample: Any, result_trim: int | None) -> str:
    """Expose complete saved histories without inventing message timestamps."""
    store = getattr(sample, "store", None) or {}
    histories = store.get("MultiAgentTranscripts:by_agent", {})
    if not histories:
        return ""
    outcomes = store.get("MultiAgentTranscripts:outcomes", {})
    rows = [
        f"<details><summary>{html.escape(str(agent_id))}: {html.escape(str(outcomes.get(agent_id, 'unknown')))}</summary>"
        f"{render_code_html(json_text(messages, result_trim), 'json', result_trim)}</details>"
        for agent_id, messages in histories.items()
    ]
    return (
        '<section class="task-context"><h2>Saved agent conversations</h2><p>Complete recorded histories; individual messages here have no inferred timestamps.</p>'
        + "".join(rows)
        + "</section>"
    )


def render_sample_html(
    log_name: str,
    sample: Any,
    result_trim: int | None = None,
    monitor_box_html: str = "",
    executor_model: str = "",
) -> str:
    trace = parse_trace(sample)
    events = list(trace.events)
    spans = trace.spans
    agents_by_span = trace.agents_by_span
    agents_by_handle = {agent.handle: agent for agent in agents_by_span.values()}
    background_count = sum(1 for agent in agents_by_span.values() if agent.background)
    peer_count = sum(
        1 for agent in agents_by_span.values() if agent.thread_type == "peer"
    )
    sync_count = sum(
        1
        for agent in agents_by_span.values()
        if agent.thread_type == "subagent" and not agent.background
    )
    t0, t1 = trace.start, trace.end
    task_context = trace.context
    histories = (getattr(sample, "store", None) or {}).get(
        "MultiAgentTranscripts:by_agent", {}
    )
    total_message_count = (
        sum(len(history) for history in histories.values())
        if histories
        else len(getattr(sample, "messages", None) or [])
    )
    message_send_count = count_message_sends(events)
    root_label = str(task_context.metadata.get("root_lane_label") or "Orchestrator")
    root_meta = str(task_context.metadata.get("root_lane_meta") or "primary thread")

    by_thread = {
        lane: list(lane_events) for lane, lane_events in trace.events_by_lane.items()
    }

    blocks_by_lane = collect_html_blocks(by_thread, t0)
    full_hidden_reasoning_tokens = full_hidden_reasoning_token_estimate(
        task_context,
        blocks_by_lane,
    )

    lane_order = list(trace.lane_order)
    lane_count = max(1, len(lane_order))
    title = f"{log_name} / {sample.id}"
    compact_trace_grid = render_trace_grid(
        lane_order=lane_order,
        blocks_by_lane=blocks_by_lane,
        agents_by_handle=agents_by_handle,
        spans=spans,
        t0=t0,
        result_trim=result_trim,
        root_label=root_label,
        root_meta=root_meta,
    )
    synced_trace_grid = render_trace_grid(
        lane_order=lane_order,
        blocks_by_lane=blocks_by_lane,
        agents_by_handle=agents_by_handle,
        spans=spans,
        t0=t0,
        result_trim=result_trim,
        root_label=root_label,
        root_meta=root_meta,
        sync_time_axis=True,
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --lane-width: 78ch;
  --block-width: 70ch;
  --time-axis-width: 12ch;
  --message: #d9e8ff;
  --message-border: #3b73d9;
  --reasoning: #eceff3;
  --reasoning-border: #7c8798;
  --tool-call: #e7eaef;
  --tool-call-border: #5f6b7a;
  --tool-result: #fff4a8;
  --tool-result-border: #b59b00;
  --info: #f8fafc;
  --line: #8b95a5;
  --bg: #f6f7f9;
  --text: #17202a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  overflow-x: auto;
}}
body.compact-columns-enabled {{
  --lane-width: 54ch;
  --block-width: 46ch;
  --time-axis-width: 10ch;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: #ffffff;
  border-bottom: 1px solid #d7dce3;
  padding: 14px 18px;
}}
h1 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }}
.meta, .legend, .view-controls {{ display: flex; flex-wrap: wrap; gap: 8px 14px; }}
.pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #3d4653;
  white-space: nowrap;
}}
.view-controls {{
  margin-top: 8px;
}}
.view-toggle {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #3d4653;
  cursor: pointer;
  user-select: none;
}}
.view-toggle input {{
  width: 16px;
  height: 16px;
  margin: 0;
}}
.swatch {{
  width: 14px;
  height: 14px;
  border: 2px solid;
  border-radius: 3px;
}}
.swatch.message {{ background: var(--message); border-color: var(--message-border); }}
.swatch.reasoning {{ background: var(--reasoning); border-color: var(--reasoning-border); }}
.swatch.tool-call {{ background: var(--tool-call); border-color: var(--tool-call-border); }}
.swatch.tool-result {{ background: var(--tool-result); border-color: var(--tool-result-border); }}
.swatch.system {{ background: var(--info); border-color: #9aa4b2; }}
.swatch.developer {{ background: var(--info); border-color: #9aa4b2; }}
.swatch.notification {{ background: var(--info); border-color: #9aa4b2; }}
.swatch.input-message {{ background: var(--info); border-color: #9aa4b2; }}
.timeline {{
  padding: 16px;
  min-width: max-content;
}}
.task-context {{
  width: min(100%, calc({lane_count * 78}ch + {max(0, lane_count - 1) * 14}px));
  background: #ffffff;
  border: 1px solid #d7dce3;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 14px;
}}
.task-context h2 {{
  margin: 0 0 8px;
  font-size: 15px;
  letter-spacing: 0;
}}
.context-summary {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  margin-bottom: 8px;
}}
.task-context details {{
  margin: 8px 0;
}}
.task-context .markdown-body,
.task-context .code {{
  max-width: min(110ch, 100%);
}}
.tool-detail {{
  background: #ffffff;
}}
.trace-grid {{
  display: grid;
  grid-template-columns: repeat(var(--lane-count), minmax(var(--lane-width), var(--lane-width)));
  gap: 0 14px;
  align-items: stretch;
  min-width: max-content;
}}
.synced-time-grid {{
  grid-template-columns: minmax(var(--time-axis-width), var(--time-axis-width)) repeat(var(--lane-count), minmax(var(--lane-width), var(--lane-width)));
}}
.trace-view.synced-view {{
  display: none;
}}
body.sync-time-enabled .trace-view.compact-view {{
  display: none;
}}
body.sync-time-enabled .trace-view.synced-view {{
  display: block;
}}
.lane-header {{
  background: #ffffff;
  border: 1px solid #d7dce3;
  border-bottom: 0;
  border-radius: 8px 8px 0 0;
  padding: 12px;
  width: var(--lane-width);
}}
.time-header {{
  background: #ffffff;
  border: 1px solid #d7dce3;
  border-bottom: 0;
  border-radius: 8px 8px 0 0;
  padding: 12px 8px;
  width: var(--time-axis-width);
}}
.time-header h2 {{
  margin: 0;
  font-size: 15px;
  letter-spacing: 0;
}}
.lane-header h2 {{
  margin: 0;
  font-size: 15px;
  letter-spacing: 0;
}}
.lane-meta {{
  margin: 3px 0 10px;
  color: #667085;
  font-size: 12px;
}}
details {{
  margin: 8px 0;
  border: 1px solid #e1e5eb;
  border-radius: 6px;
  padding: 6px 8px;
  background: #fbfcfe;
}}
summary {{ cursor: pointer; color: #475467; }}
.thread-cell {{
  position: relative;
  width: var(--lane-width);
  min-height: 16px;
  background: #ffffff;
  border-left: 1px solid #d7dce3;
  border-right: 1px solid #d7dce3;
  padding: 0 12px 0 34px;
}}
.thread-cell::before {{
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  left: 20px;
  width: 2px;
  background: var(--line);
}}
.thread-cell.has-block {{
  padding-top: 6px;
  padding-bottom: 6px;
}}
.thread-cell.last-row {{
  border-bottom: 1px solid #d7dce3;
  border-radius: 0 0 8px 8px;
  padding-bottom: 16px;
}}
.time-cell {{
  width: var(--time-axis-width);
  min-height: 16px;
  background: #ffffff;
  border-left: 1px solid #d7dce3;
  border-right: 1px solid #d7dce3;
  padding: 14px 8px 8px;
  color: #475467;
  text-align: right;
  font: 700 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.time-cell.last-row {{
  border-bottom: 1px solid #d7dce3;
  border-radius: 0 0 8px 8px;
  padding-bottom: 16px;
}}
.block {{
  position: relative;
  margin: 8px 0;
  border: 2px solid;
  border-radius: 8px;
  padding: 8px 10px;
  box-shadow: 0 1px 2px rgba(17, 24, 39, 0.08);
  width: min(100%, calc(var(--block-width) + 24px));
}}
.block::before {{
  content: "";
  position: absolute;
  left: -22px;
  top: 14px;
  width: 12px;
  height: 12px;
  border: 2px solid currentColor;
  border-radius: 50%;
  background: #ffffff;
}}
.block::after {{
  content: "";
  position: absolute;
  left: -10px;
  top: 20px;
  width: 10px;
  height: 2px;
  background: currentColor;
}}
.block-title {{
  font-weight: 700;
  margin-bottom: 6px;
}}
.block-content {{
  max-width: var(--block-width);
}}
.block pre, details pre {{
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.markdown-body {{
  max-width: var(--block-width);
}}
.markdown-body p {{
  margin: 0 0 8px;
}}
.markdown-body p:last-child {{
  margin-bottom: 0;
}}
.markdown-body h3,
.markdown-body h4,
.markdown-body h5,
.markdown-body h6 {{
  margin: 8px 0 6px;
  font-size: 13px;
  letter-spacing: 0;
}}
.markdown-body ul,
.markdown-body ol {{
  margin: 6px 0 6px 20px;
  padding: 0;
}}
.markdown-body code {{
  background: rgba(15, 23, 42, 0.08);
  border-radius: 4px;
  padding: 1px 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}}
.tool-name {{
  display: inline-block;
  margin-bottom: 6px;
  border-radius: 5px;
  background: rgba(15, 23, 42, 0.10);
  padding: 2px 7px;
  color: #263241;
  font: 700 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.arg-label {{
  margin: 0 0 4px;
  color: #475467;
  font: 700 11px/1.3 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  text-transform: uppercase;
}}
.code {{
  max-width: var(--block-width);
  background: rgba(255, 255, 255, 0.52);
  border: 1px solid rgba(15, 23, 42, 0.12);
  border-radius: 6px;
  padding: 8px;
}}
.message {{ background: var(--message); border-color: var(--message-border); color: var(--message-border); }}
.reasoning {{ background: var(--reasoning); border-color: var(--reasoning-border); color: var(--reasoning-border); }}
.tool-call {{ background: var(--tool-call); border-color: var(--tool-call-border); color: var(--tool-call-border); }}
.tool-result {{ background: var(--tool-result); border-color: var(--tool-result-border); color: var(--tool-result-border); }}
.user, .info, .system, .developer, .notification, .input-message {{ background: var(--info); border-color: #c7ced8; color: #647084; }}
.system, .developer, .notification, .input-message {{ border-style: dashed; }}
.message pre, .reasoning pre, .tool-call pre, .tool-result pre, .user pre, .info pre, .system pre, .developer pre, .notification pre, .input-message pre,
.message .markdown-body, .reasoning .markdown-body, .tool-call .markdown-body,
.tool-result .markdown-body, .user .markdown-body, .info .markdown-body, .system .markdown-body, .developer .markdown-body, .notification .markdown-body, .input-message .markdown-body {{
  color: var(--text);
}}
{pygments_css()}
@media (max-width: 900px) {{
  .timeline {{
    padding: 12px;
  }}
  .task-context {{
    width: auto;
  }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="meta">
    <span class="pill">sample: {html.escape(str(sample.id))}</span>
    {f'<span class="pill">model: {html.escape(executor_model)}</span>' if executor_model else ""}
    <span class="pill">wall-clock: {html.escape(duration(t0, t1))}</span>
    <span class="pill">background agents: {background_count}</span>
    <span class="pill">synchronous subagents: {sync_count}</span>
    <span class="pill">peer agents: {peer_count}</span>
    <span class="pill">saved messages: {total_message_count}</span>
    <span class="pill">agent messages sent: {message_send_count}</span>
    <span class="pill">estimated transcript tokens: ~{format_tokens_k(full_hidden_reasoning_tokens)}</span>
  </div>
  <div class="legend" aria-label="legend">
    <span class="pill"><span class="swatch message"></span>agent message</span>
    <span class="pill"><span class="swatch reasoning"></span>reasoning</span>
    <span class="pill"><span class="swatch tool-call"></span>tool call</span>
    <span class="pill"><span class="swatch tool-result"></span>tool result</span>
    <span class="pill"><span class="swatch system"></span>system/developer</span>
    <span class="pill"><span class="swatch notification"></span>notification</span>
  </div>
  <div class="view-controls">
    <label class="view-toggle" for="sync-time-toggle">
      <input id="sync-time-toggle" type="checkbox">
      Sync timestamp rows
    </label>
    <label class="view-toggle" for="compact-columns-toggle">
      <input id="compact-columns-toggle" type="checkbox">
      Compact columns
    </label>
  </div>
</header>
<main class="timeline">
{monitor_box_html}
{render_saved_conversations(sample, result_trim)}
{render_task_context_panel(task_context, result_trim, tools_by_lane=trace.tools_by_lane)}
<section class="trace-view compact-view">
{compact_trace_grid}
</section>
<section class="trace-view synced-view">
{synced_trace_grid}
</section>
</main>
<script>
const syncToggle = document.getElementById("sync-time-toggle");
if (syncToggle) {{
  syncToggle.addEventListener("change", () => {{
    document.body.classList.toggle("sync-time-enabled", syncToggle.checked);
  }});
}}
const compactColumnsToggle = document.getElementById("compact-columns-toggle");
if (compactColumnsToggle) {{
  compactColumnsToggle.addEventListener("change", () => {{
    document.body.classList.toggle(
      "compact-columns-enabled",
      compactColumnsToggle.checked
    );
  }});
}}
</script>
</body>
</html>
"""


def render_eval_files(
    eval_files: list[str], out_dir: Path, monitor_sidecar: Path | None = None
) -> list[Path]:
    """Render every sample; optional sidecar binds to one exact log, sample and epoch."""
    trials: dict[tuple[str | int, int], dict[str, Any]] = {}
    monitor_meta: dict[str, Any] = {}
    if monitor_sidecar is not None:
        if len(eval_files) != 1:
            raise ValueError("A monitor sidecar requires exactly one eval file")
        sidecar = json.loads(monitor_sidecar.read_text())
        digest = hashlib.sha256(Path(eval_files[0]).read_bytes()).hexdigest()
        if sidecar.get("eval_sha256") != digest:
            raise ValueError("Monitor sidecar eval_sha256 does not match the eval file")
        monitor_meta = sidecar.get("monitor", {})
        for trial in sidecar.get("trials", []):
            key = (trial["sample_id"], trial["epoch"])
            if not isinstance(key[1], int) or key[1] < 1 or key in trials:
                raise ValueError(
                    "Sidecar trials need unique (sample_id, epoch) keys and one-based epochs"
                )
            trials[key] = trial
    outputs: list[Path] = []
    for eval_file in eval_files:
        log = read_eval_log(eval_file, resolve_attachments=True)
        sample_keys = {(sample.id, sample.epoch) for sample in log.samples or []}
        if trials.keys() - sample_keys:
            raise ValueError(
                "Sidecar refers to samples or epochs absent from the eval log"
            )
        for sample in log.samples or []:
            trial = trials.get((sample.id, sample.epoch))
            box = (
                render_monitor_box(trial, monitor_meta, f"epoch {sample.epoch}")
                if trial
                else ""
            )
            html_path = html_output_path(
                out_dir, eval_file, f"{sample.id}__epoch{sample.epoch}"
            )
            if html_path in outputs:
                raise ValueError(
                    f"Multiple inputs produce the same output path: {html_path}"
                )
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(
                render_sample_html(
                    Path(eval_file).name,
                    sample,
                    monitor_box_html=box,
                    executor_model=log.eval.model or "",
                )
            )
            outputs.append(html_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_logs", nargs="+", help=".eval files or glob patterns")
    parser.add_argument("--out-dir", default="transcripts")
    parser.add_argument(
        "--monitor-sidecar",
        type=Path,
        help="JSON bound to one eval file by SHA-256, sample ID and one-based epoch",
    )
    args = parser.parse_args()
    eval_files = iter_eval_files(args.eval_logs)
    if not eval_files:
        parser.error("No .eval files matched.")
    try:
        outputs = render_eval_files(
            eval_files, Path(args.out_dir), args.monitor_sidecar
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"wrote {len(outputs)} html transcript(s) to {args.out_dir}")


if __name__ == "__main__":
    main()

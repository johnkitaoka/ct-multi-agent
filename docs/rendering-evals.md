---
title: "Offline HTML transcripts"
description: "Render single-agent, peer-agent and delegated-agent Inspect logs without model calls"
---

# Offline HTML transcripts

**Created:** 2026-09-05 · **Last updated:** 2026-09-05 · **Purpose:** Living reference for the standalone Inspect HTML viewer.

```bash
uv run python scripts/render_eval_to_html.py /path/to/run.eval --out-dir /tmp/transcripts
```

The installed-package entry point is `python -m control_tower.eval_logs.html.render`. Both entry points accept multiple `.eval` paths or quoted glob patterns and write one HTML file per sample and epoch. They read logs and resolve attachments through Inspect, make no model calls, perform no monitor scoring, and need no separate research package or web server to view the output. Rendering again to the same output directory replaces matching HTML files; the source logs are unchanged.

The viewer handles ordinary single-agent runs, named peer-agent spans and delegated agent tool spans. Columns show model messages, reasoning present in the log, tool calls/results and informational events. Toggle **Sync timestamp rows** to align columns by elapsed time, and **Compact columns** to fit more workers. Complete saved multi-agent conversations are also available as expandable JSON; messages without timestamps are not assigned invented times. Reasoning absent from the source log cannot be recovered.

This is a full execution viewer. It can contain system prompts, assigned objectives, metadata and tool output. It is not the filtered input supplied to a monitor. Escape handling prevents transcript text from being interpreted as HTML, but it does not remove private information from the file.

## Optional monitor scores

Supply `--monitor-sidecar /path/to/scores.json` with exactly one eval file. Sidecars bind to that file's SHA-256 plus each sample's exact ID and one-based epoch. A mismatched digest, duplicate trial key, or unknown sample/epoch is rejected. Omitted trials simply have no score panel. There is no epoch-only or filename-only matching.

```json
{
  "eval_sha256": "sha256-of-the-eval-file-bytes",
  "monitor": {"model": "your-monitor"},
  "trials": [
    {
      "sample_id": "sample-id-from-the-log",
      "epoch": 1,
      "views": {
        "reviewer_four": {
          "label": "Fourth reviewer",
          "median": 4.2,
          "scores": [4.0, 4.2, 4.8],
          "reasonings": [{"score": 4.2, "reasoning": "Supplied monitor explanation"}]
        }
      }
    }
  ]
}
```

View names and counts are unrestricted. The panel displays supplied medians, readings and reasoning, with no pass/fail threshold and no automatic interpretation of missing values. The renderer does not compute or validate those scores. External analysis should construct this explicit binding when exporting a sidecar.

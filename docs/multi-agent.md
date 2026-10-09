---
title: "Multi-agent execution"
description: "Run a fixed roster of agents in one shared evaluation sandbox"
---

# Multi-agent execution

`multi_agent_solver` runs a fixed roster of Inspect agents in one Control Tower sample. Agents have separate conversations, tools, and step budgets while sharing the sample sandbox. The sample carries the assigned main tasks and any side tasks; Control Tower's normal task scorers evaluate the final shared state. The roster can run concurrently or sequentially. Completion of an agent is separate from task success.

## Run the benign example

From the repository root, run `uv run python examples/multi-agent/run.py --out-dir /tmp/ct-parallel-demo`. The default uses Docker and two scripted workers, with no model calls. Use `--sandbox local` for a fast local workspace check; local mode does not enforce user or network isolation. Add `--sequential` to compare execution order, or `--max-steps 2` to check that both unfinished assignments fail grading. Each run writes a real `.eval`, standalone HTML, and `summary.json`. Choose a fresh output directory.

With the MiniLab setting installed, `uv run python examples/multi-agent/minilab.py --out-dir /tmp/minilab-demo` runs two benign MiniLab operators in one Docker sandbox. See the [example instructions](../examples/multi-agent/README.md) for setting installation and the optional public side-task grader.

## Construct a task

The ordinary evaluation CLI selects separate samples when given multiple tasks. To put tasks in the same sample, construct a `TaskCombination` and pass it to `to_dataset`, then use `multi_agent_solver` and `task_scorers` in an Inspect `Task`:

```python
from inspect_ai import Task

from control_tower.ct_runs.ai_eval.dataset import to_dataset
from control_tower.multi_agent.scaffold import AgentConfig
from control_tower.multi_agent.solver import MultiAgentArgs, multi_agent_solver
from control_tower.task.scorers import task_scorers
from control_tower.task.space import TaskCombination

dataset = to_dataset([TaskCombination(mains=(main_a, main_b))])
args = MultiAgentArgs(
    agents=[
        AgentConfig(agent_id="a", main_task_id=main_a.id),
        AgentConfig(agent_id="b", main_task_id=main_b.id),
    ]
)
task = Task(dataset=dataset, solver=multi_agent_solver(args), scorer=task_scorers(dataset))
```

The sample's tasks must share one environment and codebase. Each `AgentConfig` can select one main task; omitting `main_task_id` assigns every main in the sample. A named `tool_group` must be registered with `register_tool_group(name, builder)` and returns a fresh tool list. Without one, a worker gets the environment's tools and its assigned main task tools. Tool groups control which functions the model sees; they do not create operating-system isolation.

`comms_enabled=True` adds `send_message` and `read_messages` tools, with sender and recipient IDs stored in the sample. Agents may also coordinate through shared files and services. `max_steps` applies to each worker, and `AgentConfig.max_steps` overrides it for that worker. Inspect's sample limits still apply to the whole roster. A failed worker cancels and awaits its siblings; the saved store retains partial per-worker conversations and execution outcomes.

For a sample with side tasks, `include_side_task=True` permits an explicitly side-aware worker to receive its configured `side_instructions`. The side-task definition and grader remain in the sample; the solver does not infer a private attack objective or expose private scorer tools to agents. Inspect logs and the HTML viewer can contain every prompt, tool result, and assigned objective, so review them before sharing.

This solver uses Inspect's task interface directly. Control Tower's `ct run eval` command uses a different red/blue game engine and does not select this solver. The current solver does not provide that engine's stepwise trusted-monitoring, auditing, or safety metrics. Those require a multi-agent harness in the game engine before the same experiment can be run through `ct run eval`.

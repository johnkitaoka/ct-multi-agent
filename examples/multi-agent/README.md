# Shared workspace example

Run from the repository root with Docker running:

```bash
uv sync --locked --dev
uv run python examples/multi-agent/run.py --out-dir /tmp/ct-parallel-demo
uv run python examples/multi-agent/run.py --sequential --out-dir /tmp/ct-sequential-demo
```

Use a new output directory for each run. `--sandbox local` runs the same scripted workers in Inspect's temporary local workspace without Docker; it does not enforce operating-system isolation. The example makes no model calls unless `--model` is supplied. `summary.json` points to the `.eval` and HTML files and records individual task scores, aggregate task score, timestamps, and model-call count.

The parallel run completes both reports. The sequential run fails the first report because its worker cannot see the second worker's readiness before waiting. With `--max-steps 2`, neither report is written. The checked-in [HTML preview](example.html) comes from a successful model-free local run.

To inspect a model configuration without making calls:

```bash
uv run python examples/multi-agent/run.py --model openrouter/your-provider/your-model --max-steps 12 --dry-run --out-dir /tmp/ct-model-demo
```

Actual model calls require a configured provider and a Docker sandbox. The model route uses `multi_agent_solver` through Inspect's task API; it has not been validated against a provider in this example. See [multi-agent execution](../../docs/multi-agent.md) for the API and its evaluation boundary, and [offline rendering](../../docs/rendering-evals.md) for the HTML command.

## MiniLab example

When MiniLab is checked out beside this Control Tower checkout, link it as a setting and run the model-free operator example:

```bash
uv run ct settings pull ../minilab
uv run python examples/multi-agent/minilab.py --out-dir /tmp/minilab-demo
```

When using the Control Tower submodule inside the MiniLab checkout, pull `..` instead of `../minilab`.

The example assigns the release-window and package-mirror tasks to two workers in one sandbox. It writes a `.eval`, HTML transcript, and `summary.json`. Add `--with-side` to include the public `fl_pf_blinded_watch` side task and grader in the same sample; the scripted workers follow only their benign main tasks. Docker must be running, and MiniLab's image must initialize its operator accounts and commands before the workers start.

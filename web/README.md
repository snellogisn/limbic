# limbic web console

A tiny **local** website to drive the limbic arm in plain language and browse the
run logs. Pure Python standard library + vanilla JS — **no dependencies, no build
step**, so it starts with one command.

```bash
cd "Code Pouch/limbic"
python web/server.py                 # then open http://localhost:8765
# python web/server.py --port 9000   # pick another port
```

- **Ask page** (`/`) — type a tabletop task (or click a test button). The pipeline
  plans it, runs it on the arm, and shows the result + a link to the full log.
- **Logs page** (`/runs`) — a scrollable list of every run, newest first. Click one
  to see its **thinking** (decisions), **data** (what it sensed), and **movements**
  (what it did). Failures are flagged.

No hardware is required: with nothing plugged in, the control layer auto-selects
the software **mock** arm, so the whole site is safe and deterministic on any
laptop. (The mock runs fast here because `web/pipeline.py` shrinks the simulated
motion delays via the `LIMBIC_*_DT` env vars; real hardware timing is unchanged.)

## Two planning modes (set automatically)

| Mode | When | Who plans |
|---|---|---|
| **claude** | `ANTHROPIC_API_KEY` is set | the real brain (`limbic.brain.plan_and_run`) asks Claude to perceive + compile a plan |
| **offline** | no key | a built-in rule-based planner (handles home / move / pick / place / push) |

The offline mode is what keeps the site fully usable — including the test buttons —
without any configuration. Set a key to plan free-form instructions with Claude:

```bash
export ANTHROPIC_API_KEY=sk-...
python web/server.py
```

## The "cannot complete" scenario

Some requests can't be done, and the console shows that as a distinct outcome
(amber **cannot complete**, with the reason) rather than a crash. Two test buttons
demonstrate it:

- **Failure: impossible task** — *"make me a cup of coffee from the kitchen"*. It's
  not a tabletop pick/place/move/push, so the planner declines and explains what
  the arm *can* do.
- **Failure: out of reach** — *"move to (900, 600)"*. The point is far outside the
  arm's workspace, so it's refused with the reach limit named.

In claude mode the same outcome covers a model refusal or a run where Claude never
commits a plan. Either way nothing moves and the reason is logged to `thinking`.

## JSON API (also how Claude Code drives it)

The pages are thin clients over a small JSON API you can call directly:

```bash
# Run a task
curl -s localhost:8765/api/run -H 'content-type: application/json' \
  -d '{"task": "pick up the block at (160, 40) and place it at (160, -40)"}' | python -m json.tool

# List past runs / inspect one
curl -s localhost:8765/api/runs | python -m json.tool
curl -s localhost:8765/api/runs/<run_id> | python -m json.tool

# Force a mode regardless of the API key:
curl -s localhost:8765/api/run -H 'content-type: application/json' \
  -d '{"task": "home the arm", "mode": "offline"}'
```

Each run writes a folder under `logs/<timestamp>-<task>/` containing
`movements.jsonl`, `data.jsonl`, `thinking.jsonl`, the readable `thinking.md`,
`run.json`, and `web_result.json`.

---

## Using the arm with Claude Code (no website, no API key)

You don't need the website or an Anthropic key to drive the arm — Claude Code can
act as the planner itself, because the primitive and input catalogs are designed
to be browsed and composed:

1. **See what the arm can do and sense:**
   ```bash
   python -c "from limbic.primitives import registry; import json; print(json.dumps(registry.catalog(), indent=2))"
   python -c "from limbic.inputs import registry; import json; print(json.dumps(registry.catalog(), indent=2))"
   ```
2. **Ask Claude Code to write a plan** — a list of `{"primitive", "args"}` steps —
   for your task, using only primitives from the catalog (table frame: `+x`
   forward, `+y` left, millimetres; top-down grasps).
3. **Run the plan inside a logged run** so it's captured like any other:
   ```python
   from limbic import RobotArm, runlog
   from limbic.primitives.run_sequence import run_plan

   plan = [  # <- the list Claude Code produced
     {"primitive": "home", "args": {}},
     {"primitive": "pick", "args": {"x_mm": 160, "y_mm": 40}},
     {"primitive": "place", "args": {"x_mm": 160, "y_mm": -40}},
   ]
   with runlog.run("pick and place via claude code"):
       with RobotArm() as arm:        # mock if no hardware, real SO-101 if attached
           run_plan(arm, plan)
   ```
4. **If a better primitive is needed,** Claude Code can author one by adding a file
   to `limbic/primitives/library/` (copy an existing one as a template); the
   registry auto-discovers it.

This is the same flow the website automates — driving it from Claude Code just puts
the planning in your hands instead of the API's.

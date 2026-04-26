# Problem Statement & Design — Global Logistics Resolver (OpenEnv)

**Author:** Nikitha Thammaiah · Solo
**Hackathon themes hit:**
**Long-Horizon Planning** (primary) · **Multi-agent interactions** (secondary) · **World modeling** (tertiary).

> **What's novel:** A **hierarchical multi-agent** policy (Dispatcher + per-truck Drivers, an inter-agent message bus, and a re-plan trigger) operating over a **partially-observable, adversarially-perturbed** OpenEnv environment with **composable `openenv.core` rubrics**, feeding a **TRL + LoRA** training pipeline that consumes the same trajectories the agents produce.

---

## 1. Problem statement

Large language models are strong at single-shot reasoning but often fail when acting as **operations leads** over **many tool calls**, under **natural-language constraints**, and when **telemetry is incomplete** until they explicitly synchronize with systems (no "free look" at the whole world). They also **repeat failure modes** across episodes unless they retain compact lessons from prior graded outcomes.

This environment trains and evaluates agents that must **plan**, **follow instructions**, **probe before committing** (world modeling under partial observability), **coordinate** with peer agents through messages, and **re-plan** when the world changes mid-episode — aligned with **(Super) Long-Horizon Planning** (strategic resource management worlds), **Multi-agent interactions** (enterprise applications), and **World modeling** (belief-state tracking under uncertainty) from the hackathon themes.

---

## 2. Environment

- **Kind:** Deterministic **discrete-time logistics simulator** over a graph (warehouses, routes, trucks, orders, budget, clock).
- **OpenEnv compliance:** `LogisticsEnv` subclasses `openenv.core.Environment[LogisticsAction, LogisticsObservation, LogisticsState]`; actions and observations subclass the framework's `Action` / `Observation` Pydantic bases (`extra="forbid"` on actions). The HTTP shim exposes the canonical Gym-style API (`POST /reset`, `GET /state`, `POST /step`) plus `GET /grader` and `GET /baseline`. Validates clean: `python -m openenv.cli validate .` → "Ready for multi-mode deployment."
- **Partial observability (World modeling):** After reset, **`GET /state`** returns a **public observation**: route **statuses** and order **deadline precision** are **masked** until the agent runs **`check_network`**, which flips **full telemetry** for subsequent states. **Grading** uses the **full internal state** so success still reflects true logistics outcomes.
- **Adversarial-event mode (`ADVERSARIAL=1`):** On the first `wait` after the env has processed more than three actions, the env fires a one-shot incident: appends a `[BREAKING]` alert, **re-locks telemetry** (so `public_state()` is partial again until the next `check_network`), and on `medium`/`hard` closes `North_to_East`. In-flight trucks finish their current dispatch — the rubric is whether the agent **re-syncs and re-plans**, not whether it survives a gotcha.
- **Artifacts:** `openenv.yaml`, `env.py`, `tasks.py`, `server/app.py`, `models.py`, `inference.py`, `inference_multi.py`, `agents/`, `training/`, `notebooks/`.

---

## 3. Agent capabilities

- Parse **JSON actions** only: `check_network`, `load_truck`, `route_truck`, `wait`.
- Read a **mission brief** and **hard instruction constraints** (delivered in state).
- Operate under **partial observability** until `check_network` unlocks SLA/route truth.
- **Multi-step** plans: load → route → wait → arrivals → order fulfillment; **delayed** satisfaction when inventory and time align.
- **Hierarchical multi-agent harness** (`inference_multi.py`):
  - **Dispatcher** — reads mission brief + (partial) telemetry and emits a typed `Plan` of per-truck assignments (`{truck_id, order_id, priority, route_hint, load_amount}`) plus a `telemetry_first` flag. Does not act; only plans.
  - **Driver(T101)** / **Driver(T102)** — execute one action per turn against the OpenEnv HTTP API based on their assignment, the latest sync state, and recent peer messages. Round-robin scheduling.
  - **Inter-agent message bus** (`agents/bus.py`) — Drivers may emit a side-channel `broadcast` field alongside their action; the orchestrator strips it before posting to `/step` and replays the last few messages into each Driver's prompt. Each broadcast surfaces in logs as `[MSG] from=Tnnn body="…"`.
  - **Re-plan trigger** — when the orchestrator sees a `[BREAKING]` line in an observation, it asks the Dispatcher for a fresh plan, swaps Driver assignments, and emits a `[REPLAN]` log line. This makes the adversarial event a **coordination** challenge, not just a recovery one.
  - Same `[START]/[STEP]/[END]` log shape as the single-agent baseline (with extra `tag=multi` and `agent=…` fields) so platform validators and our plot script parse both runs uniformly.
- **Episodic memory:** Both inference harnesses append each episode's `[END]` summary to `.episode_memory.txt` (gitignored) and inject the tail into the next session's system prompt — a thin self-improvement signal usable for before/after ablations.

---

## 4. Tasks

| Task ID    | Skill tested                                                                          |
| ---------- | ------------------------------------------------------------------------------------- |
| **easy**   | Detour when a primary route is **closed**; meet SLA under time pressure.              |
| **medium** | **Budget-constrained** transfer; cost–benefit routing.                                |
| **hard**   | **Multi-order** prioritization (VIP vs standard), strike-related closures, shared budget. |

Each episode resets **task-specific** incidents, orders, and budgets via `get_task_setup`.

---

## 5. Reward model & evaluation logic

The grader is a **composition of `openenv.core.rubrics.Rubric` atoms** (the deck explicitly recommends "composable rubrics > monolithic scoring"):

| Atom                       | Signal                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| `DeliveryCompletion`       | 1.0 iff the named/VIP/standard order is fulfilled                                            |
| `TimeEfficiency`           | 1.0 if delivered within the efficient window, 0.8 by deadline, 0 if missed                   |
| `BudgetRetention`          | `max(0, budget) / cap` clipped to [0, 1]                                                     |
| `InsolvencyPenalty`        | Subtractive penalty if final budget is negative                                              |
| `NetworkSyncDiscipline`    | Multiplicative 0.85 penalty if the agent never called `check_network` (instruction-following) |

Per-task graders (`EasyRubric`, `MediumRubric`, `HardRubric`) compose these via `WeightedSum` + multiplicative gates and expose every atom's `last_score` for trainers / the eval notebook to introspect:

```python
rubric = LogisticsGrader.rubric_for("hard")
rubric(action=None, observation=final_state)
for name, child in rubric.named_rubrics():
    print(name, round(child.last_score or 0.0, 3))
```

- **Step rewards (shaping):** small signals for valid loads/routes, penalties for invalid moves, stall discouragement on `wait`, fulfillment / miss spikes — exposed as `reward.value` and `info`.
- **Terminal grader:** `LogisticsGrader.evaluate(task_level, final_state)` maps **raw** performance in \([0,1]\) to **reported scores in \((0,1)\)** (open interval) for platform / LiteLLM validators.

---

## 6. Post-training / self-improvement strategy

- **Training notebook (Colab):** [`notebooks/train_driver_trl.ipynb`](notebooks/train_driver_trl.ipynb) collects trajectories from the live env, filters by terminal grader score (`>= 0.7`), runs **SFT + LoRA** on a small `Driver` model (default Qwen2.5-0.5B-Instruct on a T4), and writes `sft_loss_curve.png` plus a **baseline vs trained** bar chart of mean `/grader` score per task. A second cell promotes the **GRPO** loop using `GET /grader` directly as the reward function (deck-recommended pattern: training loop **connects to the environment**, not a static dataset).
- **Improvement-evidence pipeline:** `python -m training.collect_and_plot --seeds N [--llm]` runs scripted, single-agent, and multi-agent rollouts, writes `docs/plots/rollout_summary.json`, and renders `docs/plots/baseline_vs_multi.png`.
- **Episodic memory tail:** Both harnesses inject the last few graded `[END]` summaries into the next system prompt; useful for before/after ablations and "self-generated data" framing from the deck.

---

## 7. Reproducibility checklist

```bash
# 1. Local server (no LLM key needed)
python -m server.app                            # http://localhost:7860/docs

# 2. Scripted-baseline plot (no LLM key needed)
python -m training.collect_and_plot --seeds 3 --out docs/plots

# 3. Multi-agent demo with surprise event (needs OPENAI_API_KEY or HF_TOKEN)
ADVERSARIAL=1 python -m server.app &            # in another shell
python inference_multi.py

# 4. End-to-end push-button demo (server + multi-agent + adversarial)
bash scripts/demo.sh

# 5. Training run (Colab T4 / HF Jobs T4-small) — produces:
#    docs/plots/sft_loss_curve.png
#    docs/plots/baseline_vs_trained_grader.png
#    docs/plots/grader_eval_summary.json
# Open notebooks/train_driver_trl.ipynb, set OPENENV_BASE_URL, Run All.
```

---
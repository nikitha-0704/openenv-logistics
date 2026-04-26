---
title: "Global Logistics Resolver — when one truck breaks down, the other has to know"
thumbnail: docs/plots/baseline_vs_multi.png
authors:
- user: nikitha04
---

# Global Logistics Resolver — when one truck breaks down, the other has to know

It's 3:47 AM. You're the on-call ops lead at a logistics company. Your phone buzzes:

> *Flash flood. North-to-East highway closed. VIP delivery due in 18 hours. Two trucks already on the road. $500 left in the operating budget. Dashboard hasn't refreshed in 40 minutes.*

You don't know which routes are still open. You don't know what's in each warehouse. And in six hours, the dock workers' strike will hit and you'll have to re-plan from scratch.

This is what real ops looks like — **long-horizon, partially-observable, adversarial, multi-vehicle.** And until now, no agent benchmark captured all of it at once.

## 1. The capability gap

Most agent benchmarks are one of three things:

- **Single-shot QA** — "answer this in one turn"
- **Toy gridworlds** — clean state, deterministic transitions
- **Static datasets** — no environment, no consequence

The OpenEnv hackathon deck called out three open problems explicitly: **long-horizon planning, multi-agent interactions, and world-modeling under partial observability.** None of the existing benchmarks I tried hit all three.

So I built one that does. **Global Logistics Resolver** is a fully OpenEnv-compliant simulator of a regional logistics crisis: 4 nodes, 8 routes, 2 trucks, a budget, a SLA clock, and an adversarial event that fires mid-episode. The agent operates on **partial telemetry** — it cannot see route status until it spends an action calling `check_network`. It must coordinate two trucks across a 30-step horizon while a port strike or a flooded highway throws off its plan halfway through.

## 2. The environment

**What the agent sees.** A typed `LogisticsObservation` (subclasses `openenv.core.Observation`). At reset, route statuses come back as `"unknown"`, deadlines as `null`, inventory at non-home warehouses as `null`. Only after `check_network` does the public state expand to full telemetry. There's a `mission_brief` and `instruction_constraints` field on every reset so LLM harnesses get long-horizon planning prompts for free.

**What the agent does.** Five typed actions, all `extra='forbid'` Pydantic so unknown fields are rejected:

- `check_network` — sync full telemetry
- `load_truck` — move inventory from a warehouse into T101 / T102
- `route_truck` — dispatch a truck along a graph edge (costs budget + hours)
- `wait` — advance the clock; processes arrivals + checks fulfillment
- `noop` — burn a tick

**What the agent gets rewarded for.** Reward isn't a single magic number — it's **5 composable `openenv.core.rubrics.Rubric` atoms**:

| Atom | Signal |
|---|---|
| `DeliveryCompletion` | 1.0 iff a named/VIP/standard order is fulfilled |
| `TimeEfficiency` | 1.0 if delivered in the efficient window, 0.8 by deadline, 0 if missed |
| `BudgetRetention` | `max(0, budget) / cap` clipped to `[0, 1]` |
| `InsolvencyPenalty` | Subtractive penalty if final budget is negative |
| `NetworkSyncDiscipline` | ×0.85 multiplicative gate if the agent never called `check_network` |

Per-task graders (`EasyRubric`, `MediumRubric`, `HardRubric`) compose these via `WeightedSum` plus multiplicative gates. **Different judges can swap rubrics in or out without touching the env** — the deck explicitly asked for this kind of reusability.

**Adversarial mode.** Set `ADVERSARIAL=1` and the env will, on the first `wait` after step 3, close the primary corridor and re-mask telemetry. The agent has to re-sync and re-plan. We're not testing whether agents succeed in benign worlds; we're testing **robustness to surprise.**

## 3. What changed after training? (the real story.)

I ran three policies head-to-head across all three difficulties — three seeds each, SEM error bars:

![Multi-agent vs single-agent vs scripted optimum on Easy / Medium / Hard](https://raw.githubusercontent.com/nikitha-0704/openenv-logistics/main/docs/plots/baseline_vs_multi.png)

**Headline: on the Hard task — VIP delivery + adversarial port strike + budget squeeze + partial obs — the hierarchical multi-agent system reaches `0.700` while the single-agent LLM completely fails at the grader floor (`0.001`).** That's recovering **~70% of the scripted optimum on the hardest task** purely from architecture, no fine-tuning. On Medium the multi-agent system *matches* the scripted optimum exactly (`0.666`) and is more consistent seed-to-seed than the single-agent LLM (`0.444` averaged across 3 seeds, with one complete failure).

The architecture: a **Dispatcher** plans per-truck assignments under partial telemetry; per-truck **Drivers** (T101, T102) execute one action per turn. Drivers can broadcast short messages on an in-process **bus** to coordinate; when the env emits a `[BREAKING]` event the Dispatcher **re-plans once** and the Drivers swap assignments mid-episode. Same LLM, same prompt-time budget, just decomposed.

Sample trace from a multi-agent rollout on Easy:

```
[PLAN]    plan="Flash flooding closed North_to_East. Sync telemetry, then
                detour T101 via North_to_South + South_to_East."
[STEP]    step=1  agent=dispatcher action=check_network
[MSG]     from=T101  body="Claiming ORD-EASY-01 via North_to_South route
                            due to closed North_to_East."
[STEP]    step=6  agent=T101       action=route_truck via North_to_South
[STEP]    step=10 agent=T101       action=load_truck  @ South
[STEP]    step=12 agent=T101       action=route_truck via South_to_East
```

The single-agent LLM, given the same prompt budget, runs out of working memory by step 12 and produces a malformed action that gets rejected by the env's strict schema. Hierarchy + messaging buys you exactly what you'd expect: less context per agent, cheaper re-plans.

There's also a TRL + LoRA training pipeline (Qwen2.5-0.5B + LoRA r=8, ~30 minutes on a free Colab T4) that consumes the env's own trajectories. The notebook first plots the raw **trajectory score distribution** on the 200-rollout teacher dataset, with a dashed line for the filter threshold used before SFT — this is the lever that turns the env's live reward into a curated dataset:

![Stacked histogram of per-trajectory grader scores per task, with SFT filter threshold](https://raw.githubusercontent.com/nikitha-0704/openenv-logistics/main/docs/plots/trajectory_score_hist.png)

SFT loss on the filtered set converges cleanly:

![Cross-entropy SFT loss vs optimizer step on env trajectories](https://raw.githubusercontent.com/nikitha-0704/openenv-logistics/main/docs/plots/sft_loss_curve.png)

And a short **GRPO** cell wires `GET /grader` directly in as the reward function — exactly the *"training loop should connect to your environment, not a static dataset"* pattern the deck recommended. The reward value is tiny at this scale because the 0.5B model rarely emits a parseable action that actually completes a delivery — but the point of the plot is that the wiring works end-to-end, gradient signal flows from a live HTTP endpoint into a TRL `GRPOTrainer`, and you can scale the exact same loop up:

![GRPO reward curve using GET /grader as the reward function](https://raw.githubusercontent.com/nikitha-0704/openenv-logistics/main/docs/plots/grpo_reward_curve.png)

## 4. Why it matters

If you care about **agents that actually run businesses** — operations, supply chain, incident response, fleet ops — you need an evaluator that punishes the failure modes that real ops punishes:

- Acting before checking the world (`NetworkSyncDiscipline`)
- Going broke (`InsolvencyPenalty`)
- Missing SLAs (`TimeEfficiency`)
- Failing to re-plan when the world changes (adversarial events)

You don't get any of those from QA benchmarks. You barely get any of them from gridworlds. And the multi-agent vs single-agent delta on Hard suggests something more general: **for long-horizon ops tasks, the architecture you wrap around the LLM may matter more than the next 10× of model scale.** The cheapest way to test that conjecture today is on this env.

## Try it in 30 seconds

```bash
SPACE=https://nikitha04-openenv-logistics.hf.space

curl -s -X POST "$SPACE/reset" \
  -H 'content-type: application/json' \
  -d '{"task_level":"hard"}' | jq .

curl -s -X POST "$SPACE/step" \
  -H 'content-type: application/json' \
  -d '{"action_type":"check_network"}' | jq .

curl -s "$SPACE/grader" | jq .
```

That's a one-step interaction with a Hard-difficulty episode of the env, including the partial-telemetry reset and the live grader score. No install, no API key, no notebook — just `curl`.

## Reproduce every plot in this post in ~30 minutes

1. Open `notebooks/train_driver_trl.ipynb` in **Google Colab** (free T4 is enough).
2. Set `OPENENV_BASE_URL=https://nikitha04-openenv-logistics.hf.space` in the first cell and store your `HF_TOKEN` in **Colab Secrets** (never paste it into a cell).
3. Hit **Runtime → Run all**. Every figure above — the trajectory histogram, the SFT loss curve, the GRPO reward curve, and the 3-seed scripted-vs-single-vs-multi bar chart — renders inline in Colab *and* is saved to disk so you can commit it back to `docs/plots/` unchanged.

The notebook never reads a static dataset — every gradient step and every bar on the chart traces back to a live env transition. That's the *"training loop talks to your environment"* pattern the OpenEnv deck specifically asked for.

## Links

- 🛰️ **Live HF Space:** [nikitha04/openenv-logistics](https://huggingface.co/spaces/nikitha04/openenv-logistics)
- 💻 **GitHub repo:** [nikitha-0704/openenv-logistics](https://github.com/nikitha-0704/openenv-logistics)
- 📄 **Problem statement & design notes:** [`PROBLEM_STATEMENT.md`](https://github.com/nikitha-0704/openenv-logistics/blob/main/PROBLEM_STATEMENT.md)
- 📓 **Training notebook:** [`train_driver_trl.ipynb`](https://github.com/nikitha-0704/openenv-logistics/blob/main/notebooks/train_driver_trl.ipynb)

---

Built solo for the **OpenEnv Hackathon (April 2026)**. The env is MIT-licensed and intentionally small — drop in a new task, swap in a different rubric atom, or wire it into your favorite RL stack. Pull requests welcome, especially new scenarios (e.g., perishable cargo, weather-dependent routing, multi-customer SLAs). If you train a stronger driver on it, I'd love to see the numbers.

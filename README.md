---
title: OpenEnv Logistics
emoji: 🚛
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: "OpenEnv logistics API: reset, state, step, grader (FastAPI)."
tags:
  - openenv
  - fastapi
  - docker
---

======================================================================
         📦 GLOBAL LOGISTICS RESOLVER v1.0.0 - DEPLOYED 📦
======================================================================
  STATUS:   [OPERATIONAL]
  GRAPH:    [SYNCED: S-N Route Active]
  GRADERS:  [DETERMINISTIC]
  LLM:      [GPT-4O-MINI VALIDATED]
  AUTHOR:   [NIKITHA THAMMAIAH / SOLO WARRIOR]
======================================================================

# 🚛 Global Logistics Resolver: Ops-Lead Simulator (OpenEnv)

> **"It's 2 AM, the N-E Highway is underwater, and the VIP client at East just called for an update. You have two trucks, a shrinking budget, and 18 hours to fix it. What’s the move?"**

The **Global Logistics Resolver** is a deterministic Reinforcement Learning (RL) environment designed to stress-test an agent's **spatial reasoning**, **SLA prioritization**, and **resource optimization** in high-stakes supply chain scenarios.

---

## 🎯 The Hook: Why This Environment?
Most LLM benchmarks focus on text generation or simple logic. This environment fills a crucial gap by forcing agents to act as **On-Call Ops Leads**. It requires:
* **Arithmetic Precision:** Managing a dwindling $300–$500 budget where every mile costs money.
* **Spatial Detouring:** Navigating graph-based networks when primary routes suffer "Cascade Failures."
* **Value-Based Logic:** Deciding whether to incur a budget penalty to save a VIP contract.

---

## 🛠️ Action & Observation Spaces
The environment strictly complies with the **OpenEnv spec** using Pydantic typing for 100% parseable interactions.

### **Action Space:**
* `check_network`: Syncs full telemetry, including route statuses and SLA countdowns.
* `load_truck`: Maneuvers inventory from warehouses into specific assets (T101/T102).
* `route_truck`: Dispatches assets via the graph. Costs budget and takes real-time hours.
* `wait`: Advances the simulation. Processes arrivals and checks for order fulfillment.

### **Observation Space:**
* **News Ticker:** Real-time status updates (e.g., `[SYSTEM_OK] Fuel prices stable.`)
* **Incident IDs:** Every crisis is tracked via unique IDs (e.g., `INC-2026-HAR-01`).
* **SLA Tracking:** Explicit feedback on how many hours remain before an order fails.

### **Reward & step metadata**
* **`reward`:** Pydantic **`LogisticsReward`** — JSON shape `{"value": <float>}` for the shaped step reward.
* **`info`:** **`RewardInfo`** — budget remaining, task completion flag, penalties (see `models.py`).

---

## 🎭 Scenarios (The Crises)
1. **Easy - Operation Flash Flood:** The primary N-E route is closed. The agent must successfully detour through the South hub. *Tests: Pathfinding.*
2. **Medium - Operation Budget Squeeze:** Stock is zero at the destination. The agent must transfer 50 units using an optimal route without exceeding a strict $300 cap. *Tests: Cost Optimization.*
3. **Hard - Operation Cascade Failure:** A regional port strike has paralyzed the South hub. The agent must manage two trucks to prioritize a VIP SLA while a secondary standard order also requires attention. *Tests: Prioritization & Parallel Processing.*

---

## 🚀 Setup & Usage

### **Evaluation Harness (Environment Variables)**
The system is built for automated evaluation. The following variables are supported:
* `HF_TOKEN` (preferred) or `OPENAI_API_KEY`: API credentials for the LLM.
* `OPENENV_BASE_URL` (or `ENV_BASE_URL`): Base URL of **this** environment’s HTTP API (default `http://localhost:7860`). Use your **Hugging Face Space URL** when the server is remote.
* `API_BASE_URL`: **OpenAI-compatible LLM base URL**. If unset and `HF_TOKEN` is set, `inference.py` defaults to **`https://router.huggingface.co/v1`**. If you use only `OPENAI_API_KEY`, omit `API_BASE_URL` to use OpenAI’s default host, or set `API_BASE_URL` explicitly (e.g. a custom gateway).
* `MODEL_NAME`: The target LLM (default `gpt-4o-mini`).

If you previously aimed `API_BASE_URL` at this **environment** only, switch that value to **`OPENENV_BASE_URL`** so `API_BASE_URL` can mean the **LLM** host when evaluators set it that way.

### **Local server (OpenEnv layout)**
1. `pip install -e .` (or `uv sync` if you use uv).
2. Start the API: **`uv run server`** or **`python -m server.app`** (ASGI app: **`server.app:app`**).
3. Optional: **`openenv validate .`** — should report **ready for multi-mode deployment**.

### **Installing Docker on macOS (no Docker Desktop password prompt)**
If `brew install --cask docker` stops at `sudo`, use **Colima** + the Docker CLI instead:
```bash
brew install colima docker
colima start          # first time downloads a VM; keep this running for docker build
docker version        # should show Client + Server
```
Ensure `/opt/homebrew/bin` is on your **`PATH`**. If `docker build` fails pulling images with TLS/certificate errors, fix network/proxy trust on your machine or try another network.

### **Running via Docker (Hugging Face Recommended)**
1. Build the image: `docker build -t logistics-resolver .`
2. Run the container: `docker run -p 7860:7860 logistics-resolver` (Hugging Face sets **`PORT`** automatically; the image respects it.)
3. Access the API documentation at `http://localhost:7860/docs`

### **Running the Baseline Agent**
1. Ensure the server is active (default `http://localhost:7860`) or set `OPENENV_BASE_URL` to your Space URL.
2. `export OPENAI_API_KEY="your_key_here"` (or `HF_TOKEN` per evaluator).
3. Optional: `export API_BASE_URL="https://..."` only if you use a custom OpenAI-compatible LLM endpoint.
4. `python inference.py`


## 📊 Baseline Performance (Validated)

| Task | Scripted Optimum | LLM Typical (GPT-4o-mini) | Logic Demonstrated |
| :--- | :---: | :---: | :--- |
| **Easy** | 1.0 / 1.0 | 1.0 / 1.0 | Identified detour route; met <14h deadline. |
| **Medium** | 0.67 / 1.0 | 0.67 / 1.0 | Optimized transfer; $100 budget retention. |
| **Hard** | 1.0 / 1.0 | 0.70 / 1.0 | Parallel truck deployment vs. VIP priority. |
> **Note:** The **Scripted optimum** column matches `test_local.py` / optimal play; **`GET /baseline`** returns these values. **LLM typical** is approximate zero-shot **GPT-4o-mini** performance.

## 📂 Project Structure
* `env.py`: The "Ops Lead" physics engine.
* `tasks.py`: Scenario definitions and deterministic graders.
* `models.py`: Pydantic schemas for the OpenEnv spec.
* `server/app.py`: FastAPI app + **`main()`** for the **`server`** console script (`uv run server`).
* `inference.py`: Autonomous OpenAI-based agent loop (submission / demo script).
* `openenv.yaml`: Metadata for the OpenEnv leaderboard.
* `pyproject.toml` / `uv.lock`: Package metadata and OpenEnv **multi-mode** validation.

Space metadata follows [Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference).

**Author:** Solo Warrior (Nikitha Thammaiah)  
**Tag:** `openenv`

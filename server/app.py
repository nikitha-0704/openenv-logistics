"""
FastAPI app for OpenHaul (OpenEnv-compatible HTTP API).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on path for env / models / tasks at project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import html
import json
from importlib.metadata import PackageNotFoundError, version as pkg_version

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from env import LogisticsEnv
from models import LogisticsAction, LogisticsReward
from tasks import LogisticsGrader, MISSION_BRIEF, SCENARIOS, open_unit_score

try:
    _VERSION = pkg_version("openhaul")
except PackageNotFoundError:
    _VERSION = "1.0.0"

_OPENAPI_TAGS = [
    {"name": "meta", "description": "Landing page, health, and build identity."},
    {"name": "environment", "description": "Episode loop: reset, observe, step (partial telemetry until `check_network`)."},
    {"name": "evaluation", "description": "Tasks, terminal grader score, and scripted baseline reference."},
]

app = FastAPI(
    title="OpenHaul — OpenEnv logistics API",
    description=(
        "Break-glass logistics under disruption: **POST /reset** → **POST /step** → **GET /grader** over HTTP, "
        "with composable rubrics and partial observability. Use [/docs](/docs) to try requests in the browser."
    ),
    version=_VERSION,
    openapi_tags=_OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
)


class _OpenHaulHeadersMiddleware(BaseHTTPMiddleware):
    """Lightweight deployment fingerprint for debugging (e.g. which build a Space is running)."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-OpenHaul-Service"] = "openhaul"
        response.headers["X-OpenHaul-Version"] = _VERSION
        return response


app.add_middleware(_OpenHaulHeadersMiddleware)

env = LogisticsEnv()
current_task_level = "easy"


_ROOT_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenHaul · OpenEnv logistics API</title>
  <style>
    :root {
      --bg0: #0b1020;
      --bg1: #1a1442;
      --card: rgba(255, 255, 255, 0.055);
      --card-border: rgba(255, 255, 255, 0.14);
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #818cf8;
      --accent2: #38bdf8;
      --code-bg: #020617;
    }
    * { box-sizing: border-box; }
    @keyframes meshDrift {
      0%, 100% { opacity: 0.55; transform: scale(1) translate(0, 0); }
      50% { opacity: 0.85; transform: scale(1.06) translate(1.5%, -1%); }
    }
    @keyframes riseIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes dashFlow {
      to { stroke-dashoffset: -48; }
    }
    @keyframes pulseGlow {
      0%, 100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.35); }
      50% { box-shadow: 0 0 12px 2px rgba(52, 211, 153, 0.2); }
    }
    html { color-scheme: dark; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      color: var(--text);
      background: linear-gradient(155deg, var(--bg0) 0%, #0f172a 38%, var(--bg1) 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      position: relative;
      overflow-x: hidden;
    }
    .mesh {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      background:
        radial-gradient(ellipse 900px 500px at 75% -5%, rgba(99, 102, 241, 0.35), transparent 55%),
        radial-gradient(ellipse 700px 420px at -5% 105%, rgba(14, 165, 233, 0.22), transparent 50%),
        repeating-linear-gradient(0deg, transparent, transparent 31px, rgba(148, 163, 184, 0.04) 31px, rgba(148, 163, 184, 0.04) 32px),
        repeating-linear-gradient(90deg, transparent, transparent 31px, rgba(148, 163, 184, 0.04) 31px, rgba(148, 163, 184, 0.04) 32px);
      animation: meshDrift 22s ease-in-out infinite;
    }
    .shell { width: 100%; max-width: 32rem; position: relative; z-index: 1; }
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 1.1rem;
      padding: 1.6rem 1.65rem 1.45rem;
      backdrop-filter: blur(16px) saturate(140%);
      box-shadow:
        0 0 0 1px rgba(129, 140, 248, 0.12) inset,
        0 28px 56px -16px rgba(0, 0, 0, 0.55);
    }
    .card > * { opacity: 0; animation: riseIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) forwards; }
    .card > *:nth-child(1) { animation-delay: 0.04s; }
    .card > *:nth-child(2) { animation-delay: 0.1s; }
    .card > *:nth-child(3) { animation-delay: 0.16s; }
    .card > *:nth-child(4) { animation-delay: 0.22s; }
    .card > *:nth-child(5) { animation-delay: 0.28s; }
    .card > *:nth-child(6) { animation-delay: 0.34s; }
    .card > *:nth-child(7) { animation-delay: 0.4s; }
    .card > *:nth-child(8) { animation-delay: 0.46s; }
    .card > *:nth-child(9) { animation-delay: 0.52s; }
    .doc-hub {
      margin: 0 0 0.9rem;
      padding: 1rem 1rem 0.95rem;
      border-radius: 0.75rem;
      background: rgba(15, 23, 42, 0.5);
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset;
    }
    .doc-hub-kicker {
      display: block;
      font-size: 0.62rem;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #94a3b8;
      margin-bottom: 0.4rem;
    }
    .doc-hub-lede {
      margin: 0 0 0.8rem;
      font-size: 0.8rem;
      line-height: 1.55;
      color: #cbd5e1;
    }
    .doc-hub-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; }
    .doc-hub .chip {
      display: inline-flex;
      align-items: center;
      gap: 0.32rem;
      padding: 0.4rem 0.78rem;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 500;
      text-decoration: none;
      color: #f8fafc;
      background: rgba(255, 255, 255, 0.055);
      border: 1px solid rgba(255, 255, 255, 0.1);
      transition: background 0.16s ease, border-color 0.16s ease, transform 0.16s ease;
    }
    .doc-hub .chip:hover {
      background: rgba(99, 102, 241, 0.22);
      border-color: rgba(165, 180, 252, 0.35);
      transform: translateY(-1px);
    }
    .doc-hub .chip .ico { font-style: normal; opacity: 0.88; font-size: 0.85em; }
    .hero-viz {
      margin: 0 0 0.75rem;
      max-width: 100%;
    }
    .hero-viz svg { width: 100%; height: auto; display: block; max-height: 5.5rem; }
    .hero-viz path.net-line {
      fill: none;
      stroke: url(#routeGrad);
      stroke-width: 1.35;
      stroke-linecap: round;
      stroke-dasharray: 7 5;
      animation: dashFlow 4.5s linear infinite;
      opacity: 0.85;
    }
    .hero-viz circle.hub { fill: #38bdf8; filter: drop-shadow(0 0 4px rgba(56, 189, 248, 0.6)); }
    .hero-viz circle.depot { fill: #a5b4fc; filter: drop-shadow(0 0 3px rgba(165, 180, 252, 0.5)); }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--accent2);
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.28);
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      margin-bottom: 0.85rem;
      width: fit-content;
    }
    .badge-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: #34d399;
      animation: pulseGlow 2.4s ease-in-out infinite;
    }
    h1 {
      margin: 0 0 0.4rem;
      display: flex;
      align-items: center;
      gap: 0.45rem;
      line-height: 1.15;
    }
    h1 .emoji { font-size: 1.5rem; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.35)); }
    h1 .brand {
      font-size: 1.5rem;
      font-weight: 800;
      letter-spacing: -0.035em;
      background: linear-gradient(115deg, #f8fafc 0%, #c7d2fe 35%, #38bdf8 95%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .sub {
      margin: 0 0 1.25rem;
      font-size: 0.9rem;
      color: var(--muted);
      line-height: 1.55;
    }
    .links { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1.2rem; }
    .links a {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0.65rem 0.85rem;
      border-radius: 0.6rem;
      text-decoration: none;
      color: var(--text);
      background: rgba(255, 255, 255, 0.035);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-left: 2px solid transparent;
      font-size: 0.875rem;
      font-weight: 500;
      transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
    }
    .links a:hover {
      background: rgba(129, 140, 248, 0.14);
      border-color: rgba(129, 140, 248, 0.32);
      border-left-color: var(--accent2);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px -8px rgba(15, 23, 42, 0.65);
    }
    .links a span.path { font-family: ui-monospace, monospace; font-size: 0.72rem; color: var(--muted); }
    .section-label {
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.4rem;
    }
    pre#health {
      margin: 0;
      padding: 0.75rem 1rem;
      border-radius: 0.55rem;
      background: linear-gradient(165deg, #020617 0%, #0f172a 100%);
      border: 1px solid rgba(148, 163, 184, 0.18);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), inset 0 -20px 40px -24px rgba(99, 102, 241, 0.12);
      font-family: ui-monospace, "Cascadia Code", monospace;
      font-size: 0.78rem;
      line-height: 1.45;
      color: #e2e8f0;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }
    footer {
      margin-top: 1.05rem;
      padding-top: 1rem;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 0.75rem;
      color: var(--muted);
    }
    footer a {
      color: var(--accent);
      text-decoration: none;
    }
    footer a:hover { text-decoration: underline; }
    @media (prefers-reduced-motion: reduce) {
      .mesh { animation: none; opacity: 0.65; }
      .hero-viz path.net-line { animation: none; stroke-dasharray: none; }
      .badge-dot { animation: none; }
      .card > * { animation: none !important; opacity: 1 !important; }
      .links a { transition: none; }
      .links a:hover { transform: none; box-shadow: none; }
    }
  </style>
</head>
<body>
  <div class="mesh" aria-hidden="true"></div>
  <div class="shell">
    <div class="card">
      <div class="hero-viz" aria-hidden="true">
        <svg viewBox="0 0 240 96" xmlns="http://www.w3.org/2000/svg" role="img">
          <defs>
            <linearGradient id="routeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#a5b4fc"/><stop offset="55%" stop-color="#818cf8"/><stop offset="100%" stop-color="#38bdf8"/>
            </linearGradient>
          </defs>
          <path class="net-line" d="M 38 48 L 118 22 L 202 48 L 118 74 Z"/>
          <circle class="hub" cx="38" cy="48" r="5.5"/>
          <circle class="depot" cx="118" cy="22" r="5"/>
          <circle class="depot" cx="202" cy="48" r="5"/>
          <circle class="depot" cx="118" cy="74" r="5"/>
        </svg>
      </div>
      <div class="badge"><span class="badge-dot"></span> Service online</div>
      <h1><span class="emoji" aria-hidden="true">🚛</span><span class="brand">OpenHaul</span></h1>
      <p class="sub">Break-glass logistics under disruption — OpenEnv-compatible FastAPI. Reset, step, and grade a multi-truck simulator over HTTP.</p>
      <div class="links">
        <a href="/docs"><span>OpenAPI &amp; try-it console</span><span class="path">/docs</span></a>
        <a href="/health"><span>Health check</span><span class="path">/health</span></a>
        <a href="/tasks?format=html"><span>Tasks &amp; action schema</span><span class="path">/tasks?format=html</span></a>
        <a href="/redoc"><span>ReDoc (alternate API docs)</span><span class="path">/redoc</span></a>
        <a href="/version"><span>Build identity</span><span class="path">/version</span></a>
      </div>
      <div class="section-label">Live response</div>
      <pre id="health">Loading…</pre>
      <section class="doc-hub" aria-label="Documentation and repository">
        <span class="doc-hub-kicker">Repository</span>
        <p class="doc-hub-lede">This page is the live API. README, training notebook (TRL + GRPO), blog, and environment design live in this Space’s Hugging Face repository — open the links below.</p>
        <div class="doc-hub-chips">
          <a class="chip" href="https://huggingface.co/spaces/nikitha04/openenv-logistics/blob/main/README.md" target="_blank" rel="noopener noreferrer"><span class="ico" aria-hidden="true">◆</span> README</a>
          <a class="chip" href="https://huggingface.co/spaces/nikitha04/openenv-logistics/blob/main/notebooks/train_driver_trl.ipynb" target="_blank" rel="noopener noreferrer"><span class="ico" aria-hidden="true">◇</span> Notebook</a>
          <a class="chip" href="https://huggingface.co/spaces/nikitha04/openenv-logistics/blob/main/Blog.md" target="_blank" rel="noopener noreferrer"><span class="ico" aria-hidden="true">▤</span> Blog</a>
          <a class="chip" href="https://huggingface.co/spaces/nikitha04/openenv-logistics/blob/main/PROBLEM_STATEMENT.md" target="_blank" rel="noopener noreferrer"><span class="ico" aria-hidden="true">◎</span> Design</a>
        </div>
      </section>
      <footer>
        JSON root for scripts: <a href="/?format=json">?format=json</a>
        · <code>POST /reset</code> · <code>POST /step</code> · <code>GET /grader</code>
      </footer>
    </div>
  </div>
  <script>
    fetch("/health").then(function (r) { return r.json(); }).then(function (j) {
      document.getElementById("health").textContent = JSON.stringify(j);
    }).catch(function () {
      document.getElementById("health").textContent = '{"error":"Could not reach /health"}';
    });
  </script>
</body>
</html>
"""

# Human-readable /tasks view (browsers). __TASK_CARDS__ / __SCHEMA__ filled by _tasks_page_html().
_TASKS_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenHaul · Tasks &amp; action schema</title>
  <style>
    :root { --bg0: #0b1020; --bg1: #1a1442; --text: #f8fafc; --muted: #94a3b8; --accent: #818cf8; --accent2: #38bdf8; --code-bg: #020617; }
    * { box-sizing: border-box; }
    @keyframes meshDrift { 0%, 100% { opacity: 0.5; transform: scale(1) translate(0,0); } 50% { opacity: 0.82; transform: scale(1.05) translate(1%, -1%); } }
    @keyframes riseIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    html { color-scheme: dark; }
    body { margin: 0; min-height: 100vh; font-family: ui-sans-serif, system-ui, sans-serif; color: var(--text);
      background: linear-gradient(155deg, var(--bg0) 0%, #0f172a 40%, var(--bg1) 100%); padding: 1.5rem; position: relative; overflow-x: hidden; }
    .mesh { position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background: radial-gradient(ellipse 800px 480px at 80% 0%, rgba(99,102,241,0.28), transparent 52%),
        radial-gradient(ellipse 600px 400px at 0% 100%, rgba(14,165,233,0.18), transparent 48%),
        repeating-linear-gradient(0deg, transparent, transparent 28px, rgba(148,163,184,0.035) 28px, rgba(148,163,184,0.035) 29px),
        repeating-linear-gradient(90deg, transparent, transparent 28px, rgba(148,163,184,0.035) 28px, rgba(148,163,184,0.035) 29px);
      animation: meshDrift 24s ease-in-out infinite; }
    .wrap { max-width: 48rem; margin: 0 auto; position: relative; z-index: 1; }
    .head-block { margin-bottom: 1.2rem; opacity: 0; animation: riseIn 0.5s cubic-bezier(0.22,1,0.36,1) 0.06s forwards; }
    nav { margin-bottom: 0.85rem; font-size: 0.875rem; }
    nav a { color: var(--accent); text-decoration: none; margin-right: 1rem; }
    nav a:hover { text-decoration: underline; color: var(--accent2); }
    .mini-net { max-width: 11rem; margin-bottom: 0.5rem; opacity: 0.9; }
    .mini-net path { fill: none; stroke: url(#tg2); stroke-width: 1.2; stroke-dasharray: 5 4; animation: dashMini 3.5s linear infinite; }
    @keyframes dashMini { to { stroke-dashoffset: -36; } }
    h1 { font-size: 1.38rem; margin: 0 0 0.25rem; letter-spacing: -0.03em;
      background: linear-gradient(115deg, #f8fafc 0%, #c7d2fe 40%, #38bdf8 100%); -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 800; }
    .lede { color: var(--muted); font-size: 0.9rem; margin: 0 0 1.25rem; line-height: 1.5; }
    .task { background: rgba(255,255,255,0.055); border: 1px solid rgba(255,255,255,0.11); border-radius: 0.85rem;
      padding: 1rem 1.1rem; margin-bottom: 0.75rem; backdrop-filter: blur(8px);
      opacity: 0; animation: riseIn 0.48s cubic-bezier(0.22,1,0.36,1) forwards; box-shadow: 0 0 0 1px rgba(129,140,248,0.06) inset; }
    .task:nth-of-type(1) { animation-delay: 0.12s; }
    .task:nth-of-type(2) { animation-delay: 0.2s; }
    .task:nth-of-type(3) { animation-delay: 0.28s; }
    .task h2 { margin: 0 0 0.5rem; font-size: 0.95rem; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.06em; }
    .scenario { margin: 0; font-size: 0.88rem; line-height: 1.5; }
    details { margin-top: 0.5rem; font-size: 0.8rem; color: var(--muted); }
    summary { cursor: pointer; color: #cbd5e1; }
    .brief { margin: 0.4rem 0 0; white-space: pre-wrap; line-height: 1.45; }
    h3.schema-head { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted);
      margin: 1.5rem 0 0.5rem; opacity: 0; animation: riseIn 0.45s ease 0.32s forwards; }
    pre.schema { margin: 0; padding: 1rem; border-radius: 0.55rem;
      background: linear-gradient(165deg, #020617 0%, #0f172a 100%);
      border: 1px solid rgba(148,163,184,0.2); font-size: 0.72rem; line-height: 1.4; overflow: auto;
      max-height: min(70vh, 36rem); white-space: pre; box-shadow: inset 0 -24px 48px -28px rgba(99,102,241,0.1);
      opacity: 0; animation: riseIn 0.5s ease 0.38s forwards; }
    .doc-hub {
      margin: 1.25rem 0 0;
      padding: 1rem 1rem 0.95rem;
      border-radius: 0.75rem;
      background: rgba(15, 23, 42, 0.5);
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset;
      opacity: 0;
      animation: riseIn 0.48s ease 0.44s forwards;
    }
    .doc-hub-kicker {
      display: block;
      font-size: 0.62rem;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #94a3b8;
      margin-bottom: 0.4rem;
    }
    .doc-hub-lede {
      margin: 0 0 0.8rem;
      font-size: 0.8rem;
      line-height: 1.55;
      color: #cbd5e1;
    }
    .doc-hub-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; }
    .doc-hub .chip {
      display: inline-flex;
      align-items: center;
      gap: 0.32rem;
      padding: 0.4rem 0.78rem;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 500;
      text-decoration: none;
      color: #f8fafc;
      background: rgba(255, 255, 255, 0.055);
      border: 1px solid rgba(255, 255, 255, 0.1);
      transition: background 0.16s ease, border-color 0.16s ease, transform 0.16s ease;
    }
    .doc-hub .chip:hover {
      background: rgba(99, 102, 241, 0.22);
      border-color: rgba(165, 180, 252, 0.35);
      transform: translateY(-1px);
    }
    .doc-hub .chip .ico { font-style: normal; opacity: 0.88; font-size: 0.85em; }
    @media (prefers-reduced-motion: reduce) {
      .mesh { animation: none; opacity: 0.6; }
      .mini-net path { animation: none; stroke-dasharray: none; }
      .head-block, .task, h3.schema-head, pre.schema, .doc-hub { animation: none !important; opacity: 1 !important; }
    }
  </style>
</head>
<body>
  <div class="mesh" aria-hidden="true"></div>
  <div class="wrap">
    <div class="head-block">
    <nav>
      <a href="/">← Home</a>
      <a href="/tasks">Raw JSON (API)</a>
      <a href="/docs">OpenAPI /docs</a>
    </nav>
    <svg class="mini-net" viewBox="0 0 120 44" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs><linearGradient id="tg2" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#818cf8"/><stop offset="100%" stop-color="#38bdf8"/></linearGradient></defs>
      <path d="M 18 22 L 60 10 L 102 22 L 60 34 Z"/>
    </svg>
    <h1>Tasks &amp; action schema</h1>
    <p class="lede">Three scenario levels for <strong>POST /reset</strong> with <code>task_level</code>. Below: one-line crisis briefs, then the full Pydantic JSON Schema for <code>LogisticsAction</code> (what <strong>POST /step</strong> accepts).</p>
    </div>
    __TASK_CARDS__
    <h3 class="schema-head">LogisticsAction — JSON Schema</h3>
    <pre class="schema">__SCHEMA__</pre>
    <section class="doc-hub" aria-label="Documentation and repository">
      <span class="doc-hub-kicker">Repository</span>
      <p class="doc-hub-lede">Same idea: API here; README, notebook, blog, and design notes in this Space on Hugging Face.</p>
      <div class="doc-hub-chips">
        <a class="chip" href="https://huggingface.co/spaces/nikitha04/openenv-logistics/blob/main/README.md" target="_blank" rel="noopener noreferrer"><span class="ico" aria-hidden="true">◆</span> README</a>
        <a class="chip" href="https://huggingface.co/spaces/nikitha04/openenv-logistics/blob/main/notebooks/train_driver_trl.ipynb" target="_blank" rel="noopener noreferrer"><span class="ico" aria-hidden="true">◇</span> Notebook</a>
        <a class="chip" href="https://huggingface.co/spaces/nikitha04/openenv-logistics/blob/main/Blog.md" target="_blank" rel="noopener noreferrer"><span class="ico" aria-hidden="true">▤</span> Blog</a>
        <a class="chip" href="https://huggingface.co/spaces/nikitha04/openenv-logistics/blob/main/PROBLEM_STATEMENT.md" target="_blank" rel="noopener noreferrer"><span class="ico" aria-hidden="true">◎</span> Design</a>
      </div>
    </section>
  </div>
</body>
</html>
"""


def _tasks_page_html(schema: dict) -> str:
    schema_pre = html.escape(json.dumps(schema, indent=2))
    cards: list[str] = []
    for tid in ("easy", "medium", "hard"):
        label = tid.upper()
        cards.append(
            '<article class="task"><h2>'
            + html.escape(label)
            + "</h2><p class=\"scenario\">"
            + html.escape(SCENARIOS.get(tid, ""))
            + '</p><details><summary>Mission brief</summary><p class="brief">'
            + html.escape(MISSION_BRIEF.get(tid, ""))
            + "</p></details></article>"
        )
    return _TASKS_PAGE_TEMPLATE.replace("__TASK_CARDS__", "\n".join(cards)).replace("__SCHEMA__", schema_pre)


@app.get("/", tags=["meta"])
async def root(request: Request):
    """Space iframe hits `/` — serve a tiny HTML shell; API under `/reset`, `/state`, `/step`, `/docs`."""
    if request.query_params.get("format") == "json":
        return {
            "status": "ok",
            "service": "openhaul",
            "version": _VERSION,
            "docs": "/docs",
            "redoc": "/redoc",
        }
    return HTMLResponse(content=_ROOT_PAGE_HTML)


@app.post("/reset", tags=["environment"])
async def reset_env(request: Request):
    """Accept empty body, `{}`, or `null` — hackathon graders may POST with no JSON body."""
    global current_task_level
    task_level = "easy"
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw)
            if isinstance(body, dict) and body.get("task_level") is not None:
                task_level = str(body["task_level"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    current_task_level = task_level
    env.reset(task_level=current_task_level)
    return {"state": env.public_state()}


@app.get("/state", tags=["environment"])
def get_state():
    return {"state": env.public_state()}


@app.get("/health", tags=["meta"])
def health_check():
    return {
        "status": "healthy",
        "incident_id": env.state().get("incident_id"),
        "task_level": current_task_level,
        "adversarial": os.environ.get("ADVERSARIAL", "0").strip() == "1",
    }


@app.get("/version", tags=["meta"])
def version_info():
    """Which build is running (useful when comparing Space logs to a git SHA)."""
    return {
        "service": "openhaul",
        "version": _VERSION,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "task_level": current_task_level,
        "adversarial": os.environ.get("ADVERSARIAL", "0").strip() == "1",
    }


@app.post("/step", tags=["environment"])
def step_env(action: LogisticsAction):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": LogisticsReward(value=reward).model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks", tags=["evaluation"])
def get_tasks(request: Request):
    """JSON for APIs (default). Use ``?format=html`` for a readable browser page with the same data."""
    schema = LogisticsAction.model_json_schema()
    if request.query_params.get("format") == "html":
        return HTMLResponse(content=_tasks_page_html(schema))
    return {
        "tasks": ["easy", "medium", "hard"],
        "action_schema": schema,
    }


@app.get("/grader", tags=["evaluation"])
def get_grader_score():
    score = LogisticsGrader.evaluate(current_task_level, env.state())
    return {"score": score}


@app.get("/baseline", tags=["evaluation"])
def run_baseline():
    # Same raw optimums as before, mapped to (0,1) exclusive via open_unit_score.
    return {
        "baseline_scores": {
            "easy": open_unit_score(1.0),
            "medium": open_unit_score(2 / 3),  # ~$100 remaining on $300 cap scripted optimum
            "hard": open_unit_score(1.0),
        }
    }


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "7860")))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server.app:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    main()

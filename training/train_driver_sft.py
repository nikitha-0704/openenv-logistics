"""
Train a small Driver policy (SFT + LoRA) on trajectories from the live OpenEnv HTTP API.

Usage (against your Hugging Face Space):
  export OPENENV_BASE_URL=https://nikitha04-openenv-logistics.hf.space
  export HF_TOKEN=hf_...   # to download the student model from Hugging Face
  python -m training.train_driver_sft

If `pip install -e ".[train]"` fails with ReadTimeoutError (large torch download), retry with a longer timeout, e.g.:
  pip install --default-timeout=1000 -e ".[train]"

Outputs (by default into docs/plots/):
  sft_loss_curve.png, baseline_vs_trained_grader.png, grader_eval_summary.json

LoRA weights are written to training/outputs/driver_lora/ (see .gitignore).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

SCRIPTED = {
    "easy": [
        {"action_type": "check_network"},
        {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 20},
        {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_South"},
        {"action_type": "wait", "hours": 5},
        {"action_type": "load_truck", "truck_id": "T101", "warehouse": "South", "amount": 20},
        {"action_type": "route_truck", "truck_id": "T101", "route_id": "South_to_East"},
        {"action_type": "wait", "hours": 8},
    ],
    "medium": [
        {"action_type": "check_network"},
        {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 50},
        {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_East"},
        {"action_type": "wait", "hours": 10},
    ],
    "hard": [
        {"action_type": "check_network"},
        {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 40},
        {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_East"},
        {"action_type": "route_truck", "truck_id": "T102", "route_id": "South_to_North"},
        {"action_type": "wait", "hours": 5},
        {"action_type": "load_truck", "truck_id": "T102", "warehouse": "North", "amount": 10},
        {"action_type": "route_truck", "truck_id": "T102", "route_id": "North_to_South"},
        {"action_type": "wait", "hours": 10},
    ],
}


def _teacher_action(task: str, step_idx: int) -> dict:
    plan = SCRIPTED.get(task, [])
    return plan[step_idx] if step_idx < len(plan) else {"action_type": "wait", "hours": 1}


def _env_ok(base: str) -> bool:
    try:
        r = requests.get(f"{base.rstrip('/')}/health", timeout=15)
        return r.status_code == 200
    except requests.RequestException:
        return False


def collect_trajectory(base: str, task: str) -> tuple[list[dict], float]:
    reset_resp = requests.post(f"{base}/reset", json={"task_level": task}, timeout=30).json()
    state0 = reset_resp.get("state", {})
    brief = state0.get("mission_brief", "")
    constraints = state0.get("instruction_constraints", [])
    samples: list[dict] = []
    step_idx = 0
    while True:
        public = requests.get(f"{base}/state", timeout=30).json().get("state", {})
        action = _teacher_action(task, step_idx)
        prompt = (
            f"Mission brief: {brief}\n"
            f"Constraints: {json.dumps(constraints)}\n"
            f"Telemetry: {json.dumps(public)}\n"
            "Reply with ONE JSON action."
        )
        samples.append({"prompt": prompt, "action": json.dumps(action, separators=(",", ":"))})
        res = requests.post(f"{base}/step", json=action, timeout=30)
        step_idx += 1
        if res.status_code != 200 or res.json().get("done"):
            break
        if step_idx > 30:
            break
    score = float(requests.get(f"{base}/grader", timeout=30).json().get("score", 0.0))
    return samples, score


def _pick_device_dtype():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _run_training(args) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(
            "Missing training deps. Install with:\n"
            "  pip install 'openhaul[train]'\n"
            "or: pip install trl transformers peft accelerate datasets torch matplotlib",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    base = args.openenv_base_url.rstrip("/")
    if not _env_ok(base):
        print(f"OpenEnv not reachable at {base}/health — start local server or set OPENENV_BASE_URL.", file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "driver_lora"
    plots_dir = Path(args.plots_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("env:", base, "| student:", args.student_model, "| device will be picked automatically")

    all_samples: list[dict] = []
    scores: list[dict] = []
    for _ in range(args.num_rollouts):
        for task in ("easy", "medium", "hard"):
            samples, score = collect_trajectory(base, task)
            scores.append({"task": task, "score": score})
            if score >= args.score_keep:
                all_samples.extend(samples)
            time.sleep(0.15)

    print("collected", len(all_samples), "training rows | score log:", scores)
    if not all_samples:
        print("No samples passed score threshold; lower --score-keep or check env.", file=sys.stderr)
        raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_chat(sample: dict) -> dict:
        messages = [
            {
                "role": "system",
                "content": "You are an autonomous logistics Driver. Output ONLY one JSON action.",
            },
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["action"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    random.shuffle(all_samples)
    ds = Dataset.from_list(all_samples).map(to_chat, remove_columns=["prompt", "action"])
    print("dataset rows:", len(ds))

    device, dtype = _pick_device_dtype()
    print("using device:", device, "| dtype:", dtype)

    # Load model: avoid device_map on MPS/CPU for broad compatibility
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        max_seq_length=1024,
        dataset_text_field="text",
        report_to="none",
    )
    if device == "cuda":
        sft_kwargs["bf16"] = True
    try:
        sft_config = SFTConfig(**sft_kwargs)
    except TypeError:
        # TRL<0.16 used `max_length` for SFTConfig in some versions
        sft_kwargs.pop("max_seq_length", None)
        sft_kwargs["max_length"] = 1024
        sft_config = SFTConfig(**sft_kwargs)

    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            args=sft_config,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            args=sft_config,
            peft_config=peft_config,
            tokenizer=tokenizer,
        )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    print("saved LoRA to", adapter_dir)

    # Loss plot
    losses = [(s["step"], s["loss"]) for s in trainer.state.log_history if "loss" in s]
    loss_path = plots_dir / "sft_loss_curve.png"
    if losses:
        xs, ys = zip(*losses)
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(xs, ys, marker="o", color="#2563eb")
        ax.set_xlabel("Optimizer step (TRL log_history)")
        ax.set_ylabel("SFT loss (cross-entropy, training batch)")
        ax.set_title(f"Driver SFT loss — {args.student_model.split('/')[-1]}")
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        fig.tight_layout()
        fig.savefig(loss_path, dpi=150)
        plt.close(fig)
        print("wrote", loss_path)
    else:
        print("no loss points in log_history (dataset very small?)", file=sys.stderr)

    if args.skip_eval:
        print("--skip-eval: done.")
        return

    def extract_action_json(text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"action_type": "wait", "hours": 1}
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {"action_type": "wait", "hours": 1}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {"action_type": "wait", "hours": 1}

    def student_rollout_grader(model_m, task: str, max_steps: int = 32) -> float:
        reset_resp = requests.post(f"{base}/reset", json={"task_level": task}, timeout=30).json()
        state0 = reset_resp.get("state", {})
        brief = state0.get("mission_brief", "")
        constraints = state0.get("instruction_constraints", [])
        model_m.eval()

        for _ in range(max_steps):
            public = requests.get(f"{base}/state", timeout=30).json().get("state", {})
            prompt = (
                f"Mission brief: {brief}\n"
                f"Constraints: {json.dumps(constraints)}\n"
                f"Telemetry: {json.dumps(public)}\n"
                "Reply with ONE JSON action."
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are an autonomous logistics Driver. Output ONLY one JSON action.",
                },
                {"role": "user", "content": prompt},
            ]
            text_in = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text_in, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model_m.device) for k, v in inputs.items()}
            with torch.no_grad():
                gen = model_m.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = gen[0, inputs["input_ids"].shape[1] :]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            action = extract_action_json(raw)
            res = requests.post(f"{base}/step", json=action, timeout=30)
            if res.status_code != 200:
                break
            if res.json().get("done"):
                break
        return float(requests.get(f"{base}/grader", timeout=30).json().get("score", 0.0))

    del trainer
    del model
    gc.collect()
    if device == "cuda":
        import torch as _t

        _t.cuda.empty_cache()

    _d, _dt = _pick_device_dtype()
    if _d == "cuda":
        base_eval = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=_dt,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    else:
        base_eval = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=_dt,
            trust_remote_code=True,
        ).to(_d)
        base_eval.eval()

    tasks_eval = ("easy", "medium", "hard")
    print("Evaluating base student …")
    baseline_scores = {t: student_rollout_grader(base_eval, t) for t in tasks_eval}
    print("baseline:", baseline_scores)

    peft_eval = PeftModel.from_pretrained(base_eval, str(adapter_dir)).eval()
    print("Evaluating base + LoRA …")
    trained_scores = {t: student_rollout_grader(peft_eval, t) for t in tasks_eval}
    print("trained:", trained_scores)

    summary = {
        "openenv_base_url": base,
        "student_model": args.student_model,
        "baseline_grader": baseline_scores,
        "trained_grader": trained_scores,
    }
    summary_path = plots_dir / "grader_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", summary_path)

    w = 0.36
    x = [0, 1, 2]
    labels = [t.capitalize() for t in tasks_eval]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(
        [i - w / 2 for i in x],
        [baseline_scores[t] for t in tasks_eval],
        width=w,
        label="Base student (pre-LoRA)",
        color="#94a3b8",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [trained_scores[t] for t in tasks_eval],
        width=w,
        label="Base + LoRA (after SFT)",
        color="#16a34a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Task difficulty")
    ax.set_ylabel("Terminal grader score (GET /grader, 0–1 open-interval)")
    ax.set_title(
        f"Baseline vs trained — {args.student_model.split('/')[-1]}\n"
        "(one JSON action per step; same harness for both)"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    bar_path = plots_dir / "baseline_vs_trained_grader.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", bar_path)


def main() -> None:
    p = argparse.ArgumentParser(description="SFT a Driver model on OpenEnv trajectories (TRL + LoRA).")
    p.add_argument(
        "--openenv-base-url",
        default=os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:7860"),
        help="OpenEnv HTTP base (default: env OPENENV_BASE_URL or local 7860)",
    )
    p.add_argument(
        "--student-model",
        default=os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    p.add_argument("--num-rollouts", type=int, default=int(os.environ.get("NUM_ROLLOUTS", "6")))
    p.add_argument("--score-keep", type=float, default=float(os.environ.get("SCORE_KEEP", "0.7")))
    p.add_argument("--output-dir", type=Path, default=Path("training/outputs/sft"))
    p.add_argument("--plots-dir", type=Path, default=Path("docs/plots"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--skip-eval", action="store_true", help="SFT + loss plot only (faster smoke test).")
    args = p.parse_args()
    _run_training(args)


if __name__ == "__main__":
    main()

"""Run tau2-bench with the MemCurator (curator_v1) memory method.

Online-memory eval loop, faithful to the SkillCurator "retrieve-before / curate-after"
contract (see ``run_unified_dev_async_curator_api.py``):

    for each task (in fixed order, across trials):
        briefing = curator.retrieve(task_text)          # BM25 top-k -> curator LLM briefing
        inject briefing into agent.system_prompt suffix  # via the patched property
        sim = run_simulation(orchestrator)               # tau2 three-party loop + evaluator
        curator.add(task_id, task_text, trajectory, reward)   # append-only write, gated by mode

The ``--memory none`` path is run by tau2's own CLI (``tau2 run``); this runner is the
``curator`` path. Both share the identical per-task orchestrator build + evaluator, so the
no-memory arm of this runner is behavior-identical to ``tau2 run`` by construction.

Results land in a SkillCurator-style folder: ``idx_<n>.json`` per (task, trial) + a
``run_config.json`` + the curator's ``curator_tau_memory.jsonl`` / ``curator_calls.jsonl``,
so the existing accuracy tooling works unchanged.

Example
-------
    export OPENAI_API_KEY=... OPENAI_BASE_URL=https://.../v1/
    python -m tau2.curator.run_curator \
        --domain airline --memory curator \
        --agent-llm gpt-4.1-2025-04-14 --user-llm gpt-4.1-2025-04-14 \
        --curation-model gpt-4.1-2025-04-14 \
        --num-trials 4 --seed 300 --curation-mode success_only_v1 \
        --exp-name airline_curator_v1
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional


def _build_config(args, task_split: Optional[str]):
    from tau2.data_model.simulation import TextRunConfig
    cfg_kwargs = dict(
        domain=args.domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=args.agent_llm,
        llm_user=args.user_llm,
        llm_args_agent={"temperature": args.agent_temperature, "max_tokens": args.max_tokens},
        llm_args_user={"temperature": args.user_temperature},
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    if task_split:
        cfg_kwargs["task_split"] = task_split
    return TextRunConfig(**cfg_kwargs)


def _task_text(task) -> str:
    """NL retrieval / curator key: the customer scenario (Q3 — natural language, not id)."""
    try:
        return str(task.user_scenario)
    except Exception:
        return task.id


def _reward_of(simulation) -> float:
    if simulation.reward_info is not None and simulation.reward_info.reward is not None:
        return float(simulation.reward_info.reward)
    return 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="tau2-bench MemCurator (curator_v1) runner")
    ap.add_argument("--domain", required=True,
                    choices=["airline", "retail", "telecom", "mock"])
    ap.add_argument("--memory", default="curator", choices=["none", "curator"],
                    help="'curator' = MemCurator online memory; 'none' = ablation (no injection).")
    ap.add_argument("--agent-llm", default="gpt-4.1-2025-04-14")
    ap.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    ap.add_argument("--curation-model", default=None,
                    help="Curator LLM. Defaults to --agent-llm.")
    ap.add_argument("--curation-base-url", default=None,
                    help="Curator base URL. Defaults to OPENAI_BASE_URL env.")
    ap.add_argument("--agent-temperature", type=float, default=0.0)
    ap.add_argument("--user-temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--num-trials", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Memory-update granularity: retrieve is frozen within a group of this "
                         "many episodes (run concurrently), curator writes fire at the group "
                         "barrier. Default = all episodes in a trial (one update per trial). "
                         "Set small (e.g. 5) to get many memory-update batches.")
    ap.add_argument("--task-split", default="base",
                    help="tau2 task split. Default 'base' matches `tau2 run` (the eval/paper split); "
                         "telecom's full set is 2285 tasks, base is 114. Pass '' for the domain default.")
    ap.add_argument("--retrieve-num", type=int, default=3)
    ap.add_argument("--curation-mode", default="success_only",
                    help="success_only | success_only_v1 | success_and_fail | success_and_fail_v1")
    ap.add_argument("--curator-on-empty", action="store_true",
                    help="Call the curator LLM even when retrieval is empty (default off = Q2).")
    ap.add_argument("--is-gateway", action="store_true",
                    help="Attach the Salesforce-gateway X-Api-Key header (from X_API_KEY env).")
    ap.add_argument("--reward-source", default="env", choices=["env", "judge"],
                    help="What GATES the curator write. 'env' (default) = tau2's ground-truth DB "
                         "reward (also the reported accuracy). 'judge' = an LLM judge verdict; the "
                         "env reward still stays the reported accuracy, only storage changes.")
    ap.add_argument("--judge-model", default=None,
                    help="Judge LLM (--reward-source judge). Defaults to --agent-llm.")
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--output-root", default="data/simulations")
    args = ap.parse_args(argv)

    from tau2.runner import build_text_orchestrator, run_simulation, get_tasks
    from tau2.curator import CuratorTau, apply_system_prompt_patch, simulation_to_text
    from tau2.curator.prompts import MEMORY_INJECTION_PREFIX, MEMORY_INJECTION_SUFFIX

    use_memory = args.memory == "curator"
    if use_memory:
        apply_system_prompt_patch()

    # Optional LLM-judge write-gate (SkillCurator --reward-source judge). The env reward stays the
    # reported accuracy; the judge only decides the reward the curator STORES with.
    use_judge = args.reward_source == "judge"
    judge_verdict = None
    if use_judge:
        from tau2.curator import judge as _judge
        from litellm import completion as _completion
        _judge_model = args.judge_model or args.agent_llm
        _judge_base = os.environ.get("OPENAI_BASE_URL")
        _judge_key = os.environ.get("OPENAI_API_KEY")

        def _judge_llm(messages):
            kwargs = dict(model=_judge_model, messages=messages, temperature=0.0,
                          max_completion_tokens=1024, num_retries=10)
            if _judge_base:
                kwargs["base_url"] = _judge_base
            if _judge_key:
                kwargs["api_key"] = _judge_key
            if args.is_gateway and os.environ.get("X_API_KEY"):
                kwargs["extra_headers"] = {"X-Api-Key": os.environ["X_API_KEY"]}
            try:
                return _completion(**kwargs).choices[0].message.content or ""
            except Exception as e:  # gpt-5.x temp rejection etc. -> retry without temperature
                if "temperature" in str(e).lower():
                    kwargs.pop("temperature", None)
                    return _completion(**kwargs).choices[0].message.content or ""
                raise

        def judge_verdict(instruction, trajectory):
            succ, score, subs, rat = _judge.judge_success(instruction, trajectory, _judge_llm)
            return {"success": succ, "score": score, "subscores": subs, "rationale": rat}

    exp_name = args.exp_name or f"{args.domain}_{args.memory}_{args.curation_mode}"
    output_path = os.path.join(args.output_root, exp_name)
    os.makedirs(output_path, exist_ok=True)

    # Curator store lives inside the result folder (self-contained run).
    curator = None
    if use_memory:
        curator = CuratorTau(
            storage_path=os.path.join(output_path, "curator_tau_memory.jsonl"),
            retrieve_num=args.retrieve_num,
            curation_model_name=args.curation_model or args.agent_llm,
            curation_base_url=args.curation_base_url or os.environ.get("OPENAI_BASE_URL"),
            curator_on_empty=args.curator_on_empty,
            curation_mode=args.curation_mode,
            is_gateway=args.is_gateway,
        )

    # Empty string --task-split '' => use the domain default (None); else the given split.
    task_split = args.task_split or None

    config = _build_config(args, task_split)

    tasks = get_tasks(args.domain, task_split_name=task_split)
    if args.num_tasks:
        tasks = tasks[: args.num_tasks]

    # Per-trial seeds derived deterministically from the run seed (so a rerun with the same
    # --seed reproduces the trial seeds). tau2's own batch runner randomizes per-trial seeds;
    # here we pin them for reproducibility of the online-memory trajectory.
    rng = random.Random(args.seed)
    trial_seeds = [rng.randint(0, 1_000_000) for _ in range(args.num_trials)]

    # Dump run config (self-describing folder).
    with open(os.path.join(output_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "runner": "tau2.curator.run_curator",
            "args": vars(args),
            "trial_seeds": trial_seeds,
            "env": {k: ("****" if "KEY" in k else os.environ.get(k))
                    for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "X_API_KEY",
                              "CURATION_TEMPERATURE", "CURATION_MAX_TOKENS", "CURATION_TOP_P")},
        }, f, indent=2, ensure_ascii=False)

    n_sims = len(tasks) * args.num_trials
    total = 0
    total_reward = 0.0
    t_start = time.time()
    _print_lock = threading.Lock()

    def _run_one(task, tseed, briefing):
        """Build + run ONE episode (LLM work, thread-safe). Retrieval/curation happen outside."""
        orchestrator = build_text_orchestrator(config, task, seed=tseed)
        if use_memory and briefing and briefing.strip():
            orchestrator.agent._system_prompt_suffix = (
                MEMORY_INJECTION_PREFIX + "\n" + briefing.strip() + MEMORY_INJECTION_SUFFIX
            )
        simulation = run_simulation(orchestrator)
        return _reward_of(simulation), simulation_to_text(simulation)

    # ORDER: trial-major (all tasks in trial 0, then trial 1, ...). Within a trial, episodes are
    # processed in GROUPS of --batch-size: memory is FROZEN within a group (retrieve for every
    # member against the pre-group store, run the group concurrently, then curator.add() all
    # members at the group barrier). This is the SkillCurator retrieve-before/curate-after
    # contract; batch_size is the memory-update granularity (small batch => many update batches).
    group_size = args.batch_size or len(tasks)
    for trial, tseed in enumerate(trial_seeds):
        for gstart in range(0, len(tasks), group_size):
            group = list(enumerate(tasks))[gstart:gstart + group_size]

            # --- retrieve BEFORE the group (frozen store snapshot, single-threaded) ---
            briefings = {}
            if use_memory:
                for ti, task in group:
                    briefings[ti] = curator.retrieve(_task_text(task), n=args.retrieve_num)

            # --- run the group's episodes CONCURRENTLY (LLM calls overlap) ---
            results = {}
            with ThreadPoolExecutor(max_workers=len(group)) as pool:
                futs = {
                    pool.submit(_run_one, task, tseed, briefings.get(ti, "")): (ti, task)
                    for ti, task in group
                }
                for fut in as_completed(futs):
                    ti, task = futs[fut]
                    try:
                        results[ti] = fut.result()
                    except Exception as e:
                        with _print_lock:
                            print(f"[task {task.id} trial {trial}] ERROR: {e}", flush=True)
                        results[ti] = (0.0, "")

            # --- persist + curate AFTER the group (barrier), in task order ---
            for ti, task in group:
                reward, trajectory = results[ti]   # reward = env ground truth (reported accuracy)
                task_text = _task_text(task)
                idx = trial * len(tasks) + ti
                out = {
                    "task_id": task.id, "trial": trial, "seed": tseed,
                    "reward": reward, "query": task_text,
                    "briefing": briefings.get(ti, ""), "trajectory": trajectory,
                }
                # Judge write-gate: env reward stays reported; the judge verdict (stored as extra
                # fields for analysis) becomes the reward the curator writes with.
                write_reward = reward
                if use_memory and use_judge and trajectory:
                    v = judge_verdict(task_text, trajectory)
                    write_reward = 1.0 if v["success"] else 0.0
                    out["env_reward"] = reward
                    out["judge_reward"] = write_reward
                    out["judge_score"] = v["score"]
                    out["judge_subscores"] = v["subscores"]
                    out["judge_rationale"] = v["rationale"]
                with open(os.path.join(output_path, f"idx_{idx}.json"), "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                if use_memory:
                    curator.add(task.id, task_text, trajectory, write_reward)
                total += 1
                total_reward += reward   # ALWAYS the env reward — accuracy unaffected by the judge

            elapsed = time.time() - t_start
            rate = total / elapsed if elapsed > 0 else 0.0
            with _print_lock:
                print(f"[{total}/{n_sims}] trial={trial} group@{gstart} "
                      f"avg={total_reward / max(total,1):.4f}  {rate * 60:.1f}/min", flush=True)

    print(f"\nFinal pass^1 (avg reward) = {total_reward / max(total, 1):.4f}  "
          f"({total_reward:.0f}/{total}) over {n_sims} sims "
          f"({len(tasks)} tasks x {args.num_trials} trials)")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()

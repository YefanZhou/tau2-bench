# tau2-bench: curator_v1 (MemCurator) — Implementation & Results

**Author:** automated run, 2026-08-11.
**Status:** baselines reproduced; curator_v1 implemented and evaluated; gpt-5.4 baselines run. Single seed (300) below; 2 more seeds (301, 302) in progress.

This document is the hand-off for review/analysis. It covers (A) what was run and the numbers,
(B) the exact result-file locations, (C) the implementation files and how they work, and
(D) important caveats.

---

## A. Results

All runs: **seed 300, num_trials 4, base task split, max_steps 200**, via the Salesforce
gateway. Metric = **pass^1** = average binary DB-state reward over all `tasks × trials` sims
(airline 50×4=200, retail 114×4=456, telecom 114×4=456). pass^1 is computed exactly as the
shipped reference files compute it (mean of per-sim `reward_info.reward`).

### A.1 Reproduction of the paper baselines — gpt-4.1 agent + gpt-4.1 user (temp 0)

| Domain  | This run (no memory) | Paper reference file | Δ vs ref |
|---------|:---:|:---:|:---:|
| airline | 0.535 | 0.560 | −0.025 |
| retail  | 0.776 | 0.741 | +0.035 |
| telecom | 0.364 | 0.342 | +0.022 |

Reference files: `data/tau2/results/final/gpt-4.1-2025-04-14_<domain>_*_4trials.json`
(airline uses `default`, telecom uses `default`, all seed 300 / 4 trials / max_steps 200).
The small gaps are consistent with gateway serving + temperature-0 tool-call nondeterminism;
config (seed, trials, max_steps) is identical. Airline is unaffected by the v1.0.1
banking_knowledge re-grade, so cross-commit comparison is valid.

### A.2 curator_v1 vs no-memory — gpt-4.1 agent + gpt-4.1 user, curator LLM = gpt-4.1

Curator config: `--batch-size 5 --curation-mode success_only_v1 --retrieve-num 3
--reward-source env`. Everything else identical to the baseline.

| Domain  | no-memory | **curator_v1** | Δ (pts) | curator store size (wins) |
|---------|:---:|:---:|:---:|:---:|
| airline | 0.535 | **0.595** | **+6.0** | 119 |
| retail  | 0.776 | **0.816** | **+4.0** | 372 |
| telecom | 0.364 | **0.575** | **+21.0** | 262 |

curator_v1 beats no-memory on **all three** subsets. Telecom (long multi-step tech-support
dialogs) benefits most.

### A.3 Capability comparison — gpt-5.4 agent + gpt-5.1 user (temp 0), no memory

| Domain  | gpt-5.4 / gpt-5.1 | gpt-4.1 / gpt-4.1 |
|---------|:---:|:---:|
| airline | 0.685 | 0.535 |
| retail  | 0.800 | 0.776 |
| telecom | 0.502 | 0.364 |

gpt-5.4 beats gpt-4.1 on every domain. Note: **not leaderboard-comparable** — gpt-5.4 is not
in the tau2 paper, and agent≠user models. gpt-5.4 accepts `temperature=0`, so tau2's default
config was used unchanged.

---

## B. Result-file locations (on GCP node `gcpssh`, host sfr-pod-...-h200-01)

Base: `~/SkillCurator-main/tau2-bench/data/simulations/`

| Run | Directory / file | Format |
|-----|------------------|--------|
| gpt-4.1 baseline airline | `airline_repro_gpt41.json/results.json` | tau2 Results |
| gpt-4.1 baseline retail  | `retail_repro_gpt41.json/results.json`  | tau2 Results |
| gpt-4.1 baseline telecom | `telecom_repro_gpt41.json/results.json` | tau2 Results |
| curator_v1 airline | `airline_curator_v1_gpt41/` | curator runner (see below) |
| curator_v1 retail  | `retail_curator_v1_gpt41/`  | curator runner |
| curator_v1 telecom | `telecom_curator_v1_gpt41/` | curator runner |
| gpt-5.4 baseline airline | `airline_gpt54_gpt51.json/results.json` | tau2 Results |
| gpt-5.4 baseline retail  | `retail_gpt54_gpt51.json/results.json`  | tau2 Results |
| gpt-5.4 baseline telecom | `telecom_gpt54_gpt51.json/results.json` | tau2 Results |

Run logs: `~/tau_repro_logs/<domain>_{repro,curator,gpt54}.log`.

**tau2 Results format** (`results.json`): top-level `simulations` (list) + `tasks` + `info`
(seed, num_trials, max_steps, git_commit). Per-sim reward at `sim.reward_info.reward`. Load
with `tau2.data_model.simulation.Results.load(path)` or read pass^1 as
`mean(s["reward_info"]["reward"] for s in json["simulations"])`.

**curator runner format** (per-run directory):
- `idx_<n>.json` — one per (task, trial), `n = trial*num_tasks + task_index`. Fields:
  `task_id, trial, seed, reward` (env DB reward = accuracy), `query` (the NL task /
  `user_scenario`), `briefing` (the injected memory, "" if none retrieved), `trajectory`
  (`[USER]/[AGENT]/[TOOL]` text). With `--reward-source judge`, adds `env_reward`,
  `judge_reward`, `judge_score`, `judge_subscores`, `judge_rationale`.
- `curator_tau_memory.jsonl` — the append-only store (`task_id, query, trajectory, reward,
  status`). Line count = number of stored wins.
- `curator_calls.jsonl` — every read-time curation call: `query, retrieved` (per-entry
  `store_index, score, rank, question, status`), `retrieved_text`, `messages` (full
  system+user prompt sent to the curator LLM), `briefing`, `briefing_raw`. Use this to audit
  retrieval quality and prompt behavior.
- `run_config.json` — args + trial seeds + masked env.

pass^1 for a curator run = `mean(json.load(idx_i)["reward"] for all idx files)`.

---

## C. Implementation

All code committed to the **tau2-bench git submodule** (`github.com/YefanZhou/tau2-bench.git`,
branch `main`, commit `857e883`), pulled to GCP. Package: `src/tau2/curator/`.

| File | Purpose |
|------|---------|
| `curator.py` | `CuratorTau`: the memory object. `add(task_id, task, trajectory, reward)` — append-only, reward-gated write (no LLM). `retrieve(query, n)` — BM25 top-k over stored task texts → build curator prompt → curator LLM → strip `<think>` → return briefing. Q2 rule: empty store → `""` with no LLM call. Curation via `litellm` (reads `OPENAI_API_KEY`/`OPENAI_BASE_URL`), `CURATION_*` env knobs, gpt-5.x temperature fallback, gateway `X-Api-Key`. |
| `bm25.py` | Vendored dependency-free `BM25Okapi` (k1=1.5, b=0.75, same idf flooring as `rank-bm25`). Used because `rank-bm25` is only a tau2 *knowledge* extra and the shared env must not be mutated. |
| `prompts.py` | 4 curation-mode system prompts (`success_only`, `success_only_v1`, `success_and_fail`, `success_and_fail_v1`) + `MEMORY_INJECTION_PREFIX/SUFFIX`. Prompts are tau2-domain-aware (describe the `[USER]/[AGENT]/[TOOL]` transcript, tool-ordering / user-confirmation / policy-check extraction) with guardrails: never override policy, never copy concrete IDs/prices/dates. Modeled on the ALFWorld/WebShop curator prompts. |
| `judge.py` | LLM-as-a-judge write-gate (`--reward-source judge`). Fig-13/15-style: 3 sub-scores `task_completion / policy_adherence / communication`, `score = mean`, `success = score ≥ 0.5`. Never raises. The env reward always stays the reported accuracy; the judge only decides what the curator stores. |
| `patch.py` | Idempotent monkey-patch of `LLMAgent.system_prompt` to append `_system_prompt_suffix` after `</policy>`. Zero-diff when no suffix set (verified against vendored LLMAgent at 668d3bc — `SYSTEM_PROMPT`/`AGENT_INSTRUCTION` still module-level; `system_prompt` a `@property`). |
| `trajectory.py` | `SimulationRun.messages` → `[USER]/[AGENT]/[TOOL]` text (tool calls via `to_functional_format`, tool outputs capped 500 chars). |
| `run_curator.py` | The runner: `python -m tau2.curator.run_curator`. Group-batched retrieve-before / curate-after loop (batch = memory-update granularity). |

### How the online-memory loop works (`run_curator.py`)
Trial-major order. Within a trial, episodes are processed in **groups of `--batch-size`**:
1. **Retrieve before the group** (single-threaded, frozen store snapshot): a briefing per group member.
2. **Run the group concurrently** (`ThreadPoolExecutor`): build `orchestrator` via
   `build_text_orchestrator(config, task, seed)`, set `agent._system_prompt_suffix` to the
   briefing, `run_simulation(orchestrator)`, read `reward_info.reward` + trajectory.
3. **Curate after the group** (barrier): `curator.add(...)` for each member (in task order),
   gated by `--reward-source` (env DB reward by default; judge verdict if set).

This is the exact "retrieve-before / update-after-group" contract from SkillCurator's
`run_unified_dev_async_curator_api.py`; `--batch-size` controls how many memory-update batches
you get (batch 5 → airline 10 batches/trial, retail/telecom 23).

### Key CLI knobs (`python -m tau2.curator.run_curator --help`)
`--domain {airline,retail,telecom,mock}` `--memory {curator,none}` `--agent-llm` `--user-llm`
`--curation-model` (default = agent-llm) `--num-trials` `--seed` `--batch-size`
`--task-split base` (default; telecom full set = 2285 tasks, base = 114) `--curation-mode`
`--retrieve-num` `--reward-source {env,judge}` `--judge-model` `--is-gateway` `--exp-name`.

### Reproduce a curator run (GCP)
```bash
cd ~/SkillCurator-main/tau2-bench
set -a && . ../.env && set +a
export OPENAI_API_KEY=$GATEWAY_OPENAI_API_KEY OPENAI_BASE_URL=$GATEWAY_OPENAI_API_BASE
./.venv/bin/python -m tau2.curator.run_curator \
  --domain airline --memory curator \
  --agent-llm gpt-4.1-2025-04-14 --user-llm gpt-4.1-2025-04-14 \
  --curation-model gpt-4.1-2025-04-14 \
  --num-trials 4 --seed 300 --batch-size 5 --curation-mode success_only_v1 \
  --exp-name airline_curator_v1_gpt41
```

### Reproduce a baseline (tau2 native CLI)
```bash
./.venv/bin/tau2 run --domain airline \
  --agent-llm gpt-4.1-2025-04-14 --user-llm gpt-4.1-2025-04-14 \
  --num-trials 4 --seed 300 --save-to airline_repro_gpt41.json
```

---

## D. Caveats for the analyst

1. **Single seed.** All A.1–A.3 numbers are seed 300 only. curator_v1's success-only store is
   self-reinforcing (store is built from the run's own successes; store size == success
   count), so run-to-run variance is inherent — report **mean±std over ≥3 seeds**. Seeds 301
   and 302 are being run for exactly this.
2. **Curator vs baseline comparability.** airline/retail: the runner's task set (`base`)
   equals the tau2 CLI default set with identical task ids, so curator_v1 and the baseline are
   directly comparable. telecom: the runner defaults to `--task-split base` (114 tasks) to
   match the baseline; do NOT compare against a full-set (2285-task) telecom run.
3. **The env reward is always ground truth.** With `--reward-source judge`, the judge only
   changes what the curator *stores*; the reported pass^1 is always tau2's DB-state reward.
4. **gpt-5.4/5.1 runs are an internal capability check**, not leaderboard numbers.
5. **Curator model.** Here the curator LLM = the agent LLM (gpt-4.1). Swappable via
   `--curation-model` (e.g. a trained curator, gemini, gpt-5.x).

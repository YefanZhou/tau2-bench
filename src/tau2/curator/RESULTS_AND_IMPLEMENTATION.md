# tau2-bench: memory methods (curator_v1 / ReasoningBank / SkillOS) — Implementation & Results

**Updated:** 2026-08-11. Hand-off doc for review/analysis. Covers (A) protocol, (B) results,
(C) result-file locations, (D) implementation files, (E) important corrections & caveats.

> **Two corrections since the first draft — read these first:**
> 1. **Leakage fix.** The original runs used `num_trials=4` with ONE persistent memory bank
>    across trials. tau-bench repeats each task verbatim every trial and the retrieval key is
>    the verbatim `user_scenario`, so trials 2-4 retrieved a task's OWN prior-trial winning
>    trajectory = answer leakage (violates the pass^k i.i.d. assumption). **All results below
>    use `num_trials=1`** (one lifetime: bank starts empty, accumulates across tasks within the
>    single pass, a task can only ever retrieve *other* earlier tasks). The old nt=4 numbers are
>    retired.
> 2. **gpt-5.4 temperature.** gpt-5.x models only accept `temperature=1` via litellm
>    (`UnsupportedParamsError` otherwise). tau2 sets `litellm.drop_params=True` (silently drops
>    the requested temp) and the curator has a temp-fallback (retries without temp), so **every
>    gpt-5.4 role — agent OR curator — actually ran at temperature 1.0**, regardless of the
>    requested value recorded in `run_config.json`. gpt-4.1 roles are unaffected (agent/user 0,
>    curator 0.7). Consequence: any gpt-5.4-vs-gpt-4.1 curator comparison carries a temperature
>    confound (1.0 vs 0.7) that cannot be removed on the gpt-5.4 side.

---

## A. Protocol

- Benchmark: tau2-bench (τ²), base task split. **airline = 50 tasks, retail = 114, telecom = 114.**
- **num_trials = 1** → 50 / 114 / 114 sims per run (one attempt per task; no verbatim repeat → no memory leakage).
- Metric: **pass^1** = mean per-sim binary reward = tau2's ground-truth DB-state check (not an LLM judge). At k=1 this equals plain accuracy.
- Seed 300, max_steps 200. Runs via the Salesforce gateway (gpt-5.4 agent-only runs can use the direct-OpenAI profile; both serve the same models).
- Temperatures: gpt-4.1 agent 0, gpt-4.1 user 0, gpt-4.1 curator 0.7; **all gpt-5.4 roles forced 1.0** (see correction #2).
- Memory loop (`run_curator.py`): trial-major, episodes processed in groups of `--batch-size` (=memory-update granularity). Retrieve is frozen within a group (retrieve-before), episodes run concurrently, curator writes at the group barrier (curate-after). batch-size 5 → airline 10 update-batches, retail/telecom 23.

---

## B. Results (seed 300, num_trials=1, pass^1)

### B.1 Baseline reproduction vs paper reference (gpt-4.1 agent + gpt-4.1 user)

| Domain  | our nt=1 baseline | paper reference (4-trial) | paper reproduced (our 4-trial) |
|---------|:---:|:---:|:---:|
| airline | 0.580 | 0.560 | 0.535 |
| retail  | 0.737 | 0.741 | 0.776 |
| telecom | 0.386 | 0.342 | 0.364 |

Reference = shipped `data/tau2/results/final/gpt-4.1-2025-04-14_<domain>_*_4trials.json`. All
three arms agree within a few points (single-trial noise + gateway serving). pass^1 throughout.

### B.2 Full comparison — curator model × agent model × retrieve-num

All curator runs use curation-mode `success_only_v1` unless noted; batch-size 5.

**gpt-4.1 (temp 0) as agent, gpt-4.1 (temp 0) as user simulator:**

| Curator model | rn | airline (50) | retail (114) | telecom (114) |
|---|:--:|:--:|:--:|:--:|
| none (baseline) | – | 0.580 | 0.737 | 0.386 |
| gpt-4.1 (temp 0.7) | 3 | 0.560 | 0.798 | 0.561 |
| gpt-5.4 (temp 1.0) | 3 | 0.480 | 0.772 | 0.483 |
| gpt-5.4 (temp 1.0) | 5 | 0.460 | 0.789 | 0.623 |

**gpt-5.4 (temp 1.0) as agent, gpt-4.1 (temp 0) as user simulator:**

| Curator model | rn | airline (50) | retail (114) | telecom (114) |
|---|:--:|:--:|:--:|:--:|
| none (baseline) | – | 0.640 | 0.798 | 0.491 |
| gpt-5.4 (temp 1.0) | 3 | 0.780 | 0.886 | 0.746 |
| gpt-5.4 (temp 1.0) | 5 | 0.660 | 0.877 | 0.702 |

**gpt-5.4 (temp 1.0) as agent, gpt-5.1 (temp 1.0) as user simulator** (direct-OpenAI profile):

| Curator model | rn | airline (50) | retail (114) | telecom (114) |
|---|:--:|:--:|:--:|:--:|
| none (baseline) | – | 0.640 | 0.798 | 0.491 |
| gpt-5.4 (temp 1.0) | 3 | 0.640 | 0.904 | 0.684 |
| gpt-5.4 (temp 1.0) | 5 | 0.660 | 0.886 | 0.737 |

Note vs the gpt-4.1-user block above: the **no-memory baselines are identical** (0.640 / 0.798 /
0.491) — swapping the user simulator gpt-4.1→gpt-5.1 did not move baseline accuracy on this seed.
Curator arms differ with no consistent direction (gpt-5.1-user higher on retail, lower on
airline-rn3 / telecom-rn3), and the biggest gap (airline-rn3 0.640 vs 0.780) is within the
temp-1.0 single-run variance band — so the user-simulator swap does not materially change the
conclusions at one seed. gpt-5.1 on the DIRECT OpenAI endpoint accepts temp 0/1 (unlike the
gateway's litellm path, which forces the gpt-5 family to 1); it was run explicitly at temp 1.0.

### B.3 Key findings
- **The gpt-5.4 curator flips sign with agent strength.** With the weak gpt-4.1 agent it *hurts*
  airline (0.58 → 0.48); with the strong gpt-5.4 agent the same curator *helps* airline (0.64 →
  0.78). Consistent with "memory hurts when retrieval precision is low and base competence is
  high; helps when the agent can exploit rich briefings."
- **gpt-5.4 agent + gpt-5.4 curator is the best cell on every domain** (airline 0.78, retail
  0.89, telecom 0.75 at rn3), and memory clearly helps the strong agent everywhere.
- **Why gpt-5.4 curator hurt the gpt-4.1 agent on airline** (trajectory analysis, rn3): 7 tasks
  flipped win→loss vs the gpt-4.1 curator; **3 (tasks 15, 45, 22) the no-memory baseline solved
  alone** — memory did net harm. Not length/leakage (briefing medians ~equal, ~0 id leakage);
  it's *over-confident, speculative, prescriptive* briefings the weak agent defers to. Verbatim:
  - task 15: "if it is basic economy, changes generally aren't allowed unless policy/tooling supports upgrading first" (asserts an ungrounded rule; task needed *cheapest*, agent then picked non-cheapest)
  - task 45: "it shows insurance cannot be added to an existing reservation post-booking … do not promise exceptions, vouchers, or partial refunds"
  - task 22: "if not, do not improvise … prefer one of Omar's gift cards as payment"
- **retrieve-num 3 vs 5**: no consistent winner; differences are within per-run variance.

### B.4 Prompt A/B (airline, gpt-4.1 agent, gpt-5.4 curator temp 1.0, rn3)
| Curation mode | pass^1 |
|---|:--:|
| `success_only_v1` (current) | 0.620 |
| `success_only_v2_grounded` (new, grounded/non-directive) | 0.560 |

**Inconclusive.** The same v1 prompt scored 0.480 in one run and 0.620 in another (14-pt swing
from temp-1.0 curator sampling alone), which exceeds the 6-pt A/B gap. `v2_grounded` *does*
produce the intended output (briefings ~2.6× shorter, grounded process-hints, no invented
policy rules) but a single 50-task run at temp 1.0 cannot resolve the effect. Needs ≥3 seeds/arm.

---

## C. Result-file locations (GCP node `gcpssh`, under `~/SkillCurator-main/tau2-bench/data/simulations/`)

| Run family | directory pattern |
|---|---|
| gpt-4.1 baseline | `<domain>_base_nt1_s300.json/results.json` |
| gpt-4.1 curator | `<domain>_cur_nt1_s300/` |
| gpt-4.1 agent + gpt-5.4 curator | `<domain>_cur_gpt54cur_rn{3,5}_nt1_s300/` |
| gpt-5.4 agent (no memory) | `<domain>_gpt54agent_nt1_s300.json/results.json` |
| gpt-5.4 agent + gpt-5.4 curator | `<domain>_g54agent_g54cur_rn{3,5}_nt1_s300/` |
| airline prompt A/B | `airline_ABtest_success_only_{v1,v2_grounded}_g54cur_s300/` |

Logs: `~/tau_repro_logs/*.log`. **tau2 Results** dirs have `results.json` (read pass^1 as
`mean(s.reward_info.reward)`). **curator runner** dirs have per-(task,trial) `idx_<n>.json`
(`reward`, `query`, `briefing`, `trajectory`), `curator_tau_memory.jsonl` / `reasoning_bank.jsonl`
/ `skillos_skills.json` (the store), `curator_calls.jsonl` (full curator prompt + briefing per
retrieval — the audit trail), and `run_config.json`. pass^1 = `mean(idx_i["reward"])`.

### C.1 Exact number → source-file map (every cell in the §B tables)
Substitute `<domain>` ∈ {airline, retail, telecom}. Each row's pass^1 = the value the run
printed as `Final pass^1` in its log (also recomputable from the files as noted).

| §B row | source directory (under data/simulations/) | log (under ~/tau_repro_logs/) |
|---|---|---|
| gpt-4.1 baseline | `<domain>_base_nt1_s300.json/results.json` | `<domain>_base_nt1_s300.log` |
| gpt-4.1 curator (rn3) | `<domain>_cur_nt1_s300/` | `<domain>_cur_nt1_s300.log` |
| gpt-4.1 agent + gpt-5.4 curator rn3 | `<domain>_cur_gpt54cur_rn3_nt1_s300/` | `<domain>_cur_gpt54cur_rn3_s300.log` |
| gpt-4.1 agent + gpt-5.4 curator rn5 | `<domain>_cur_gpt54cur_rn5_nt1_s300/` | `<domain>_cur_gpt54cur_rn5_s300.log` |
| gpt-5.4 agent, no memory | `<domain>_gpt54agent_nt1_s300.json/results.json` | `<domain>_gpt54agent_nt1_s300.log` |
| gpt-5.4 agent + gpt-5.4 curator rn3 | `<domain>_g54agent_g54cur_rn3_nt1_s300/` | `<domain>_g54agent_g54cur_rn3_s300.log` |
| gpt-5.4 agent + gpt-5.4 curator rn5 | `<domain>_g54agent_g54cur_rn5_nt1_s300/` | `<domain>_g54agent_g54cur_rn5_s300.log` |
| gpt-5.4 agent + gpt-5.1 user, no memory | `<domain>_g54agent_u51_base_nt1_s300.json/results.json` | `<domain>_g54agent_u51_base_s300.log` |
| gpt-5.4 agent + gpt-5.1 user + gpt-5.4 curator rn3 | `<domain>_g54agent_u51_g54cur_rn3_nt1_s300/` | `<domain>_g54agent_u51_g54cur_rn3_s300.log` |
| gpt-5.4 agent + gpt-5.1 user + gpt-5.4 curator rn5 | `<domain>_g54agent_u51_g54cur_rn5_nt1_s300/` | `<domain>_g54agent_u51_g54cur_rn5_s300.log` |
| airline A/B v1 | `airline_ABtest_success_only_v1_g54cur_s300/` | `airline_ABtest_success_only_v1_s300.log` |
| airline A/B v2_grounded | `airline_ABtest_success_only_v2_grounded_g54cur_s300/` | `airline_ABtest_success_only_v2_grounded_s300.log` |

Regenerate every number from the raw logs in one line (on the GCP node):
```bash
grep -H "Final pass\|Pass\^1 " ~/tau_repro_logs/*_s300.log ~/tau_repro_logs/*_nt1_s300.log
```
Or recompute a curator run's pass^1 directly from its idx files:
```bash
python -c "import json,glob; v=[json.load(open(f))['reward'] for f in glob.glob('data/simulations/<dir>/idx_*.json')]; print(sum(v)/len(v), len(v))"
```

---

## D. Implementation (all in `tau2-bench/src/tau2/curator/`, committed to fork `YefanZhou/tau2-bench` main)

| File | Purpose |
|------|---------|
| `curator.py` | `CuratorTau` — read-time memory curator. `add()` append-only reward-gated write; `retrieve()` BM25 top-k → curator-LLM briefing → strip `<think>`. Q2: empty store → "" no LLM. |
| `reasoningbank.py` | `ReasoningBankTau` — LLM distills ≤3 markdown memory items per episode (success/fail reflection prompts); BM25 read = concatenate items (no read LLM). Port of `reasoningbank_alfworld_api.py`. |
| `skillos.py` | `SkillOSTau` — evolving skill library via native tool-calling curation (`new_skill_insert`/`skill_update`/`skill_delete`); BM25 read over skill title+YAML-description. Port of SkillOS `skills_memory` + curation loop; self-contained (no embeddings/numpy). |
| `prompts.py` | Curator system prompts + injection wrapper. Modes: `success_only`, `success_only_v1` (default used in all runs), **`success_only_v2_grounded`** (new: process-hints only, no invented policy rules, explicit precedence, relevance-gate + length cap), `success_and_fail[_v1]`. |
| `judge.py` | LLM-judge write-gate (`--reward-source judge`): subscores task_completion / policy_adherence / communication. Env reward stays reported accuracy; judge only gates storage. |
| `patch.py` | Monkey-patch `LLMAgent.system_prompt` to append `_system_prompt_suffix` after `</policy>` (zero-diff when unset). |
| `trajectory.py` | `SimulationRun.messages` → `[USER]/[AGENT]/[TOOL]` text. |
| `bm25.py` | Vendored dependency-free `BM25Okapi` (rank-bm25 is only a tau2 knowledge extra; shared env not mutated). |
| `run_curator.py` | Runner. `--memory {none,curator,reasoningbank,skillos}`; identical retrieve/add seam + injection wrapper across the three methods. |

**Run recipe (GCP):**
```bash
cd ~/SkillCurator-main/tau2-bench
set -a && . ../.env && set +a
export OPENAI_API_KEY=$GATEWAY_OPENAI_API_KEY OPENAI_BASE_URL=$GATEWAY_OPENAI_API_BASE
export CURATION_MAX_TOKENS=8192   # REQUIRED for a gpt-5.x curator (reasoning tokens eat the cap → empty briefing otherwise)
./.venv/bin/python -m tau2.curator.run_curator \
  --domain airline --memory curator \
  --agent-llm gpt-4.1-2025-04-14 --user-llm gpt-4.1-2025-04-14 \
  --curation-model gpt-5.4 \
  --num-trials 1 --seed 300 --batch-size 5 --retrieve-num 3 \
  --task-split base --curation-mode success_only_v1 \
  --exp-name airline_cur_gpt54cur_rn3_nt1_s300
```
Baseline / gpt-5.4-agent runs use tau2's native CLI: `./.venv/bin/tau2 run --domain <d>
--agent-llm <m> --user-llm gpt-4.1-2025-04-14 --num-trials 1 --seed 300 --task-set-name <d>
--save-to <name>.json`.

The two other memory methods swap `--memory curator` for `--memory reasoningbank` or
`--memory skillos` (same seam; ReasoningBank/SkillOS smoke-tested live but not yet in the
results table above).

---

## E. Caveats
1. **Single seed (300)** everywhere → no error bars. gpt-5.4 at forced temp 1.0 shows large
   per-run variance (airline 0.78 vs 0.66 across rn on 50 tasks). Do NOT over-read retrieve-num
   or small cross-cell gaps. Multi-seed (301/302) is the planned next step.
2. **Temperature confound** in gpt-5.4-vs-gpt-4.1 curator (1.0 vs 0.7) — unavoidable (gpt-5.4
   can't run at 0.7). The `run_config.json` records the *requested* temp, not the executed one.
3. **curator vs baseline is a controlled comparison** (identical except the injected briefing).
   **gpt-5.4-agent is NOT** — it changes the agent model *and* forces temp 1.0; treat as a rough
   capability reference, not a clean ablation. Not leaderboard-comparable.
4. The env DB reward is always the reported accuracy; the LLM judge (when used) only gates writes.

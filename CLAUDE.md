# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this codebase does

A prototyping environment for running and comparing neuro-symbolic diagnostic assistants. Each *diagnostic scenario* involves three agents acting on a small toy electrical system: a **Saboteur** that injects a fault, a **ServiceAgent** that executes diagnostic actions, and a **DiagnosticAssistant** that guides the service agent toward the root cause.

## Running a scenario

```bash
python -m run_diagnostic_scenario --help
```

Typical invocations (run from the repo root):

**3-cubes system, LLM assistant, CLI:**
```bash
python -m run_diagnostic_scenario \
  --text-input-file "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_description.txt" \
  --diagram "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_schematics.png" \
  --kg "Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl" \
  --ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
  --retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/3_cubes" \
  --system 3CubesSystem --saboteur Human --service Human --assistant LLM --interface cli
```

**10-cubes system, neuro-symbolic assistant:**
```bash
python -m run_diagnostic_scenario \
  --text-input-file "Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_description.txt" \
  --diagram "Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_schematics.png" \
  --kg "Knowledge_sources/Structured_knowledge_sources/10_cubes/zorro-ontology-10-cubes-abox.ttl" \
  --ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
  --retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/10_cubes" \
  --system 10CubesSystem --saboteur Human --service Human --assistant EvidenceKGOptimal --interface cli
```

**Voice interface** (requires running uvicorn separately):
```bash
uvicorn voice_server:app --host 0.0.0.0 --port 8000
# Then browse to http://127.0.0.1:8000/client.html
```

**Batch runs:**
```bash
python -m run_many_scenarios
```

## Running the evaluation protocol

```bash
python -m run_evaluation_protocol --help
```

Runs the Adaptive Evaluation Protocol across scenarios, collecting batches of trajectories until behavior converges. Checkpoints are saved under `Logs/Evaluation/<AssistantType>/<run_id>/`.

**Typical invocation (Ollama local server):**
```bash
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
python -m run_evaluation_protocol \
  --scenarios 1,26,51,76,101,126,135 \
  --assistant EvidenceKGOptimal --service SpiceSim \
  --assistant-config '{"model":"llama3.2","embed_model":"nomic-embed-text"}' \
  --service-config '{"model":"llama3.2"}' \
  --labeling-model llama3.2 \
  --batch-size 3 --rounds 5 --max-batches 2 --max-concurrent 3
```

Key flags: `--all-scenarios`, `--batch-size`, `--rounds`, `--max-batches`, `--max-concurrent` (set to 1–3 for Ollama, ~10 for OpenAI), `--resume <run_id>`.

**Important:** When using Ollama, the Agents SDK must also be routed through it. `run_diagnostic_scenario.py` reads `OPENAI_BASE_URL` at startup and calls `agents.set_default_openai_client()` automatically.

**Monitoring progress:**
```bash
tail -f Logs/Evaluation/<AssistantType>/<run_id>/protocol.log
```

## Generating evaluation tables

```bash
python make_tables.py --run EvidenceKGOptimal:<run_id> --run LLM:<run_id>
# Auto-detect latest run per assistant:
python make_tables.py
```

Produces three outputs (printed to stdout, LaTeX copied to clipboard if `pyperclip` is installed):
1. **Detailed per-run table** — from `checkpoint.json`: N trajectories, clusters, ARI, success rate with CI, cost, actions
2. **Cross-assistant comparison table** — from `final_report.txt`: success rate, cost, actions side-by-side
3. **System-level aggregate table** — mean ± std per system

`Evaluation/make_latex_tables.py` is deprecated; use `make_tables.py` instead.

## Running tests

```bash
pytest Tests/
# Run a single test file:
pytest Tests/test_scenario1_spice.py -s
```

## Architecture

### Core abstractions (`environment_classes.py`)

All agent roles are abstract base classes that extend `ThingThatLogs` (sets up a per-class logger backed by a shared file handler from `Configuration`):

- `Saboteur.sabotage(system)` — injects a fault, returns an optional `RootCauseDescription`
- `ServiceAgent.execute_action()`, `.verify_hypothesis()`, `.decide_finish()` — executes suggested actions and decides when to stop
- `DiagnosticAssistant.setup()`, `.suggest_action()`, `.record_action_outcome()` — the core loop; `suggest_action()` returns either a `DiagnosticAction`, a `DiagnosticFaultHypothesis`, or `None`

The orchestrator (`run_diagnostic_scenario()` in `environment_classes.py`) drives the sabotage → initial observations → diagnostic loop → summary lifecycle. It is async throughout.

### Configuration (`configuration.py`)

`Configuration` is a dataclass holding all runtime parameters. It is built from CLI args by `parse_configuration()` in `run_diagnostic_scenario.py` and passed to every agent. Key fields: `SABOTEUR_TYPE`, `SERVICE_TYPE`, `ASSISTANT_TYPE`, `ASSISTANT_CONFIG`, `SERVICE_CONFIG`, `SABOTEUR_CONFIG` (JSON dicts passed via `--assistant-config` / `--service-config` / `--saboteur-config`), model names, logging paths, and RAG parameters.

Model name fields (`DEFAULT_LLM_MODEL`, `DEFAULT_NS_MODEL`, `DEFAULT_SERVICE_MODEL`, `EMBED_MODEL`) are fallback defaults only — each agent reads its actual model from its own config dict first (e.g. `ASSISTANT_CONFIG.get("model", configuration.DEFAULT_LLM_MODEL)`).

### Implementations (`Implementations/`)

Concrete agent classes; selected at runtime by the `--saboteur`, `--service`, `--assistant` CLI flags:

| Flag value | Class | Notes |
|---|---|---|
| `LLM` | `DiagnosticAssistantLLM` | Monolithic GPT agent |
| `EvidenceKGOptimal` | `DiagnosticAssistantEvidenceKGOptimal` | Neuro-symbolic: uses OWL reasoning + SPARQL + information-gain heuristic |
| `Mock` | `DiagnosticAssistantMock` | No-op, for testing |
| `Human` | `SaboteurHuman` / `ServiceAgentHuman` | Reads from CLI or voice |
| `FixedScenario` | `SaboteurFixedScenario` | Reads fault from `SCENARIOS` list |
| `SpiceSim` | `SaboteurSpiceSim` / `ServiceAgentSpiceSim` | Uses `diagnosable_systems_simulation` SPICE backend |

`Implementations/scenarios.py` defines the central `SCENARIOS` list (dataclass `Scenario` with `id`, `system_name`, `root_cause`, `fault_fns`). Scenario faults are injected via `diagnosable_systems_simulation.actions.fault_actions`.

### Neuro-symbolic assistant (`diagnosticAssistantEvidenceKGOptimal.py`)

Pipeline per `suggest_action()` call:
1. SPARQL queries on the ABox KG identify candidate problem nodes.
2. OWL reasoning (`Utilities/OWL_reasoning.py` via HermiT) expands entailments.
3. A `HeuristicTestingProcedure` (min-entropy / information-gain) selects the next diagnostic action.
4. Small LLM calls handle entity extraction and binary outcome classification.
5. When confidence is sufficient, emits a `DiagnosticFaultHypothesis` instead of an action.

### Knowledge sources (`Knowledge_sources/`)

Five physical systems are supported (all simulatable via SPICE):

| System | `--system` value | Subdirectory | Scenarios |
|---|---|---|---|
| 3-module cube circuit | `3CubesSystem` | `3_cubes/` | 1–25 |
| 10-module cube circuit | `10CubesSystem` | `10_cubes/` | 26–50 |
| Asymmetric chains | `AsymmetricChainsSystem` | `asymmetric_chains/` | 51–75 |
| Ambient light sensor | `AmbientLightSensorSystem` | `ambient_light_sensor/` | 76–105 |
| Current sensor | `CurrentSensorSystem` | `current_sensor/` | 106–135 |

Each system has:
- A textual description (`*_description.txt`)
- A schematic image (`*_schematics.png`) passed as a vision input
- An ABox KG (`zorro-ontology-*-abox.ttl`) with component instances
- The shared TBox ontology at `Structured_knowledge_sources/zorro-ontology-tbox.ttl`

The ontology namespace is `http://www.example.org/zorro/` (constant `ZORRO` in `configuration.py`).

### Utilities (`Utilities/`)

- `caching.py` — SQLite-backed LLM call cache (`Cache/cache.db`); enabled with `--cache`
- `retrieval.py` — RAG: chunk, embed (`text-embedding-3-*`), top-k retrieval from `--retrieval-folder`
- `topology.py` — Graph algorithms over the KG (minimal dense sets used by the NS assistant)
- `OWL_reasoning.py` — Wraps HermiT reasoner via `owlready2`
- `chat_log.py` — Writes human-readable HTML session logs to `Logs/Chats/`
- `multi_layer_belief_model.py` — Belief state utilities
- `agents_boilerplate.py` — Shared OpenAI Agents SDK setup

### Logging

Every run writes two outputs under `Logs/`:
- `DebuggingLogs/DIAGNOSTIC_SCENARIO_RUN_<timestamp>` — structured log (all agents share one file handler)
- `Chats/DIAGNOSTIC_SCENARIO_RUN_<timestamp>_CHAT.html` — human-readable HTML chat transcript

### Voice interface

`voice_server.py` (FastAPI + Whisper STT + TTS) serves `static/client.html`. `voice_client.py` provides `send_prompt()` / `get_user_text()` used by `VoiceHumanIO` in `environment_classes.py`. The server requires `cert.pem`/`key.pem` for HTTPS when used from a mobile device.

## Key dependencies

- `openai` (v2) + `agents` (OpenAI Agents SDK) — LLM calls and agent orchestration
- `rdflib` — KG loading and SPARQL queries
- `owlready2` — OWL reasoning via HermiT
- `diagnosable_systems_simulation` — SPICE-based circuit simulation backend (separate package)
- `pydantic` — All data models in `environment_classes.py`
- `tiktoken` — Tokenization for RAG chunking
- `pandas` — Used by `make_tables.py` for detailed table formatting (optional; falls back to plain text)
- `pyperclip` — Auto-copy LaTeX to clipboard in `make_tables.py` (optional)

## Known simulation behaviors (not bugs)

- **Non-convergence in feedback-loop scenarios**: The ambient-light-sensor system can create a genuine oscillating feedback loop (lamp → sensor → relay → lamp). SPICE non-convergence is the physically correct answer in these cases. Relay coil terminals both near rail voltage (~11.85 V) during oscillation is also correct.
- **Non-convergence with shorted lamp + current-sensing relay**: A shorted lamp draws enough current to trip the relay, which cuts power, which drops current below threshold, which re-closes the relay. This produces a genuine limit cycle — non-convergence is correct.
- **Floating-node voltage**: A disconnected port will read the voltage driven by other sources through the rest of the network — not zero. This is correct SPICE physics, not a bug.
- **Loose connection anomaly reporting**: A cable with a `LooseConnectionCoupling` reports "has a loose connection (port is intermittently disconnected)" when its port is floating during a simulation. A hard-disconnected cable (via `DisconnectCable`) reports "has a disconnected end (port is floating)". These are distinct and intentional.

## ServiceAgentSpiceSim repair tracking

`_repaired_comp_ids` tracks components genuinely fixed mid-session so the post-batch circuit reset (`restore_snapshot`) leaves them in the repaired state. It is populated by:
- Successful `replace_component` actions
- Successful `reconnect_cable` actions

The post-batch reset gates success on `result.converged` — a non-converged simulation is never treated as a successful repair.

## LLM Assistant notes

**Conversation history design (do not change without reading this):**
The conversation passed to the LLM is a flat sequence of user-role messages: system description + diagram first, then symptoms, then action outcomes. This is intentional:
- `Agent.instructions` only accepts a string — images cannot be placed there, so the system description must stay as a user message.
- Adding assistant turns between user messages would double token count with no new information: each action outcome already contains the `DESCRIPTION` field from the assistant's prior suggestion, so the model is not reasoning blind.
- The flat structure handles pre-session observations naturally — outcomes injected before the assistant starts fit seamlessly without requiring strict user/assistant turn alternation.
- Interleaving assistant turns would cause the model to re-read its own prior suggestions as first-person decisions, anchoring it to earlier hypotheses. In the flat structure, prior suggestions appear only as the `DESCRIPTION` field in outcome messages — framed as "an action was taken", a weaker anchor that leaves more room for hypothesis revision.

**Component targeting in diagnostics:**
- The LLM assistant must reference components found in the system description and schematics (e.g., "battery", "the main lamp", "the control LED", "the load diode").
- Do NOT give the LLM internal simulation component IDs (`main_bulb`, `psu_green_resistor`, `load_cable_pos`, etc.). These are implementation details of the SPICE simulator.
- Enclosures/cubes ARE valid targets for diagnostic actions: e.g., "invert the power supply cube", "rotate the control module", "open the load module inspection panel".
- The LLM should be guided to name components and enclosures as they appear in the natural language system description, not by invented IDs.

## Ollama integration notes

Set `OPENAI_BASE_URL=http://localhost:11434/v1` and `OPENAI_API_KEY=ollama` to route all LLM calls through a local Ollama server. Both the sync `OpenAI()` client (embeddings) and the Agents SDK runner are routed automatically via startup code in `run_diagnostic_scenario.py`.

- Chat model: `llama3.2:latest`
- Embedding model: `nomic-embed-text:latest` (pass as `--assistant-config '{"embed_model":"nomic-embed-text"}'`)
- Vision: `llama3.2` is text-only; pass `--assistant-config '{"no_vision":true}'` for the LLM assistant
- Parallelism: use `--max-concurrent 1–3` to avoid overloading Ollama

**Known limitation:** `llama3.2` cannot perform the symptom→component NER step that `EvidenceKGOptimal` relies on, so NS assistant trajectories will almost always be zero-cost surrenders with this model.

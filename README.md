# A Prototyping Environment for Comparing Neuro-symbolic Diagnostic Assistants

Codebase associated with the ESWC 2026 demo paper *"A Prototyping Environment for Comparing Neuro-symbolic Diagnostic Assistants"*.

The environment runs **diagnostic scenarios**: a fault is injected into a small simulated electrical system by a *Saboteur* agent, and a *Diagnostic Assistant* guides a *Service Agent* toward identifying the root cause. The goal is to compare different assistant strategies — LLM-based, neuro-symbolic, and baselines — under controlled conditions.

![Environment architecture](environment_architecture.png)

---

## Repository structure

```
ESWC_2026_Demo/
├── environment_classes.py          # Abstract agent roles and scenario orchestrator
├── configuration.py                # Configuration dataclass and defaults
├── run_diagnostic_scenario.py      # Entry point: run a single diagnostic scenario
├── run_evaluation_protocol.py      # Entry point: run the Adaptive Evaluation Protocol
├── run_many_scenarios.py           # Batch runner for multiple scenarios
├── make_tables.py                  # Aggregate evaluation results into tables
├── SCENARIOS_MASTER.csv            # Master list of all 135 scenarios with metadata
│
├── Implementations/                # Concrete agent implementations
│   ├── diagnosticAssistantLLM.py           # Monolithic LLM assistant
│   ├── diagnosticAssistantEvidenceKGOptimal.py  # Neuro-symbolic assistant
│   ├── diagnosticAssistantRandomSearch.py  # Random baseline (components one-by-one)
│   ├── diagnosticAssistantFixedRandomTrajectories.py  # Fixed random baseline
│   ├── diagnosticAssistantUnhelpful.py     # Always-wrong baseline
│   ├── fault_injections.py         # SCENARIOS list (135 injectable faults)
│   ├── saboteur*.py                # Saboteur implementations
│   └── serviceAgent*.py            # Service agent implementations
│
├── Evaluation/                     # Adaptive Evaluation Protocol
│   ├── protocol.py                 # Main protocol loop (HDBSCAN + ARI convergence)
│   ├── clustering.py               # Intent & execution clustering
│   ├── qualitative.py              # Qualitative rubric evaluation (LLM-based)
│   ├── report.py                   # Per-scenario report formatting
│   ├── metrics.py                  # Numerical metrics (success rate, cost, actions)
│   └── checkpoint.py               # Checkpoint save/load
│
├── Knowledge_sources/
│   ├── Structured_knowledge_sources/   # OWL ontology (TBox) + per-system ABox KGs
│   └── Unstructured_knowledge_sources/ # Textual descriptions + circuit schematics
│
├── Utilities/                      # Shared helpers
│   ├── OWL_reasoning.py            # HermiT reasoner wrapper (owlready2)
│   ├── retrieval.py                # RAG: chunking, embedding, top-k retrieval
│   ├── topology.py                 # Graph algorithms over the KG
│   ├── caching.py                  # SQLite-backed LLM call cache
│   ├── chat_log.py                 # HTML chat transcript writer
│   ├── agents_boilerplate.py       # OpenAI Agents SDK setup helpers
│   └── ...
│
├── Tests/                          # pytest test suite
├── static/client.html              # Browser voice interface
├── voice_server.py                 # FastAPI voice server (Whisper STT + TTS)
└── voice_client.py                 # Voice client helpers
```

---

## Systems

Five simulated electrical systems are supported, all implemented as SPICE circuits via the [`diagnosable-systems-simulation`](https://github.com/kataph/diagnosable-systems-simulation) package:

| System | `--system` flag | Scenarios |
|---|---|---|
| 3-module cube circuit | `3CubesSystem` | 1–25 |
| 10-module cube circuit | `10CubesSystem` | 26–50 |
| Asymmetric chains | `AsymmetricChainsSystem` | 51–75 |
| Ambient light sensor | `AmbientLightSensorSystem` | 76–105 |
| Current sensor | `CurrentSensorSystem` | 106–135 |

---

## Assistant implementations

| `--assistant` flag | Class | Description |
|---|---|---|
| `LLM` | `DiagnosticAssistantLLM` | Monolithic GPT agent with RAG and vision |
| `EvidenceKGOptimal` | `DiagnosticAssistantEvidenceKGOptimal` | Neuro-symbolic: OWL reasoning + SPARQL + information-gain heuristic |
| `RandomSearch` | `DiagnosticAssistantRandomSearch` | Random baseline: hypothesises components one-by-one without replacement |
| `FixedRandomTrajectories` | `DiagnosticAssistantFixedRandomTrajectories` | Fixed random action sequences (reproducible baseline) |
| `Unhelpful` | `DiagnosticAssistantUnhelpful` | Always-wrong baseline |

The neuro-symbolic assistant pipeline is summarised below:

![Neuro-symbolic assistant architecture](assistant_architecture.png)

---

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is a heavy dependency pulled in by `sentence-transformers`. If you only want to run scenarios without the evaluation protocol, you can skip it.

### 2. Install the simulation backend

The SPICE simulation backend is a separate package:

```bash
git clone https://github.com/kataph/diagnosable-systems-simulation
pip install -e diagnosable-systems-simulation/
```

### 3. Set API credentials

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI, or
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"  # any OpenAI-compatible endpoint
export OPENAI_API_KEY="sk-or-..."       # OpenRouter key
```

---

## Running a scenario

```bash
python -m run_diagnostic_scenario --help
```

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

---

## Running the evaluation protocol

The Adaptive Evaluation Protocol collects batches of trajectories per scenario and checks for convergence of the trajectory clustering (via inter-batch ARI).

```bash
python -m run_evaluation_protocol --help
```

**Example (automated, all scenarios):**
```bash
python -m run_evaluation_protocol \
  --all-scenarios \
  --assistant EvidenceKGOptimal --service SpiceSim \
  --service-config '{"model":"openai/gpt-4.1-mini"}' \
  --assistant-config '{"model":"openai/gpt-4.1","embed_model":"openai/text-embedding-3-small"}' \
  --labeling-model openai/gpt-4.1-mini \
  --batch-size 10 --rounds 10 --max-concurrent 5
```

**Incremental runs (manual convergence control):**
```bash
# First batch
python -m run_evaluation_protocol --scenarios 1,2,3,4,5 \
  --assistant EvidenceKGOptimal --service SpiceSim \
  [flags] --batch-size 10 --max-batches 1
# note the run_id printed at startup

# Add another batch
python -m run_evaluation_protocol --scenarios 1,2,3,4,5 \
  --resume <run_id> [flags] --batch-size 10 --max-batches 2

# Add more scenarios to the same run
python -m run_evaluation_protocol --scenarios 6,7,8,9,10 \
  --resume <run_id> [flags] --batch-size 10 --max-batches 1
```

**Monitor progress:**
```bash
tail -f Logs/Evaluation/<AssistantType>/<run_id>/protocol.log
```

---

## Generating evaluation tables

```bash
# Auto-detect latest run per assistant:
python make_tables.py

# Specify runs explicitly:
python make_tables.py --run EvidenceKGOptimal:<run_id> --run LLM:<run_id>
```

Produces tables broken down by scenario, system, and fault type, including qualitative analysis aggregates. LaTeX is auto-copied to clipboard if `pyperclip` is installed.

---

## Running tests

```bash
pytest Tests/
pytest Tests/test_scenario1_spice.py -s   # single file with output
```

---

## Logging

Each run produces two outputs under `Logs/`:
- `DebuggingLogs/DIAGNOSTIC_SCENARIO_RUN_<timestamp>` — structured log
- `Chats/DIAGNOSTIC_SCENARIO_RUN_<timestamp>_CHAT.html` — human-readable HTML transcript

Evaluation protocol outputs go to `Logs/Evaluation/<AssistantType>/<run_id>/`.

---

## Video

A demo video is available in the [releases](https://github.com/kataph/Diagnostic-Assistant-Demo/releases/tag/v1.0) and on [YouTube](https://youtu.be/beh18vLPm30).

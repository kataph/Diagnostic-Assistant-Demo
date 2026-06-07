import argparse
import asyncio
import logging

import rdflib

import agents
# to prevent tracing errors when using a proxy
agents.set_tracing_disabled(True)

from configuration import Configuration
from Implementations import DiagnosticAssistantEvidenceKGOptimal, DiagnosticAssistantLLM, DiagnosticAssistantMock, SaboteurHuman, ServiceAgentHuman, ServiceAgentMock, SaboteurLLMFaultTree, ServiceAgentLLM, SaboteurFixedScenario, SaboteurSpiceSim, ServiceAgentSpiceSim, DiagnosticAssistantRandomTrajectory, ServiceAgentSpiceSimMockNL
from environment_classes import SystemDescription, run_diagnostic_scenario


def parse_configuration() -> Configuration:
    parser = argparse.ArgumentParser(
        description="Initialization of diagnostic scenarios")

    # Required Positional Arguments

    parser.add_argument("--saboteur", type=str, default="Human",
                        help="Type of saboteur agent (['Human', 'LLMFaultTree', 'SpiceSim', 'FixedScenario'])")
    parser.add_argument("--service", type=str, default="Human",
                        help="Type of service agent (['Human', 'LLM', 'SpiceSim', 'Mock'])")
    parser.add_argument("--assistant", type=str, default="EvidenceKGOptimal",
                        help="Type of diagnostic assistant agent (['LLM', 'EvidenceKGOptimal', 'Mock'])")

    parser.add_argument("--text-input-file", type=str,
                        help="Text file containing a description of the system")
    # parser.add_argument("text_input", type=str, help="Primary text description of the system")
    # parser.add_argument("--output_dir", type=str, default="Output", help="Directory for output files")

    # Optional Paths
    parser.add_argument("--ontology", type=str, default=None,
                        help="Path to the ontology file")
    parser.add_argument("--kg", type=str, default=None,
                        help="Path to the kg file")
    parser.add_argument("--system", type=str, default="3CubesSystem",
                        help="Terminal part of the whole-system IRI in the ontology file [3CubesSystem, 10CubesSystem, AmbientLightSensorSystem, CurrentSensorSystem, AsymmetricChainsSystem]")
    parser.add_argument("--namespace", type=str,
                        default="http://www.example.org/zorro/", help="Namespace IRI of the ontology")
    parser.add_argument("--diagram", type=str, default=None,
                        help="Path to a graphical description of the system")

    # Retrieving
    parser.add_argument("--retrieval-folder", type=str, default=None,
                        help="Path to the folder with the documents for retrieval")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Number of top chunks to be consedered during retrieval")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Number of tokens of each chunk for retrieval")
    parser.add_argument("--chunk-overlap", type=int, default=0,
                        help="Number of tokens over which consecutive chunks overlap, for retrieval")
    parser.add_argument("--retrieving-cache", type=str,
                        default="embeddings_cache.pkl", help="Cache file for the embeddings")
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-large",
                        help="Embedding model name for retrieval")
    parser.add_argument("--tokenizer-model", type=str,
                        default="cl100k_base", help="Tokenizer model name for retrieval")

    # Neurosymolic assistant model
    parser.add_argument("--NS-assistant-model", type=str, default="gpt-4.1",
                        help="Model name for the Nneurosymbolic diagnostic assistant (it is used only for limited entity extraction and binary classification tasks)")

    # Diagnostic assistant model
    parser.add_argument("--LLM-assistant-model", type=str, default="gpt-4.1",
                        help="Model name for the LLM-monolithic diagnostic assistant")

    # Service agent model
    parser.add_argument("--LLM-service-model", type=str, default="gpt-4.1",
                        help="Model name for the LLM-monolithic servige agent")

    # Fixed saboteur scenario
    parser.add_argument("--forced-scenario", type=int, default=0,
                        help="Scenario id when using fixed scenario saboteur.")

    # Logic & Performance Flags
    parser.add_argument("--cache", action="store_true",
                        help="Enable global caching")
    # parser.add_argument("--no-cache-agents", nargs="*", default=[], help="List of agents to exclude from cache")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Max number of diagnostic rounds (suggestion-action steps)")
    # parser.add_argument("--batch-size", type=int, default=10, help="Size of the batch run")

    # Modes & Interface
    parser.add_argument("--interface", choices=["cli", "voice"], default="voice",
                        help="Interface mode (currently affects on ServiceAgentHuman class)")
    # parser.add_argument("--tester", action="store_true", help="Enable human tester mode")
    # parser.add_argument("--symptom-gen", action="store_true", help="Enable human symptom generation")

    parser.add_argument("--log-path", type=str, default="Logs/DebuggingLogs",
                        help="Relative address of the folder for log files")
    parser.add_argument("--chat-path", type=str, default="Logs/Chats",
                        help="Relative address of the folder for chat files")
    parser.add_argument("--trajectory-path", type=str, default="Logs/Trajectories",
                        help="Relative address of the folder for trajectory JSON files")
    parser.add_argument("--log-level", type=int, default=20,
                        help="Sets the loggers level (default is INFO = 20)")

    args = parser.parse_args()

    # Instantiate the dataclass with parsed arguments
    return Configuration(
        SABOTEUR_TYPE=args.saboteur,
        SERVICE_TYPE=args.service,
        ASSISTANT_TYPE=args.assistant,
        TEXT_INPUT_FILE=args.text_input_file,
        # OUTPUT_DIRECTORY=args.output_dir,
        SYSTEM_NAME=args.system,
        ONTOLOGY_PATH=args.ontology,
        KG_PATH=args.kg,
        DIAGRAM_PATH=args.diagram,
        USE_CACHE=args.cache,
        # HUMAN_TESTER=args.human_tester,
        # HUMAN_SYMPTOM_GENERATOR=args.human_symptom_generator,
        # BATCH_RUN_SIZE=args.batch_size,
        # DIAGNOSTIC_ACTIONS_BATCH_SIZE=args.diag_batch_size,
        INTERFACE_MODE=args.interface,
        # AGENTS_FORCED_TO_NO_CACHE=args.no_cache_agents

        LOG_PATH=args.log_path,
        CHAT_PATH=args.chat_path,
        TRAJECTORY_PATH=args.trajectory_path,
        LOG_LEVEL=args.log_level,
        MAX_NUMBER_OF_ROUNDS=args.rounds,
        ONTOLOGY_NAMESPACE=rdflib.Namespace(args.namespace),
        SYSTEM_URL=rdflib.URIRef(base=rdflib.Namespace(
            args.namespace), value=args.system),

        RETRIEVAL_FOLDER_PATH=args.retrieval_folder,
        TOP_K=args.top_k,
        CHUNK_SIZE=args.chunk_size,
        CHUNK_OVERLAP=args.chunk_overlap,
        CACHE_PATH=args.retrieving_cache,
        EMBED_MODEL=args.embed_model,
        TOKENIZER_MODEL=args.tokenizer_model,

        LLM_ASSISTANT_MODEL=args.LLM_assistant_model,
        NS_ASSISTANT_MODEL=args.NS_assistant_model,
        SERVICE_MODEL=args.LLM_service_model,
        FORCED_SCENARIO_ID=args.forced_scenario,

    )


configuration = parse_configuration()


def get_vision_diagram(file_path, client) -> tuple[str | None, str | None]:
    """
    Return (file_id, image_b64) for the given diagram path.

    Tries to upload the file to the OpenAI Files API first (preferred: the
    uploaded file is cached on the server and reused across turns).  If that
    fails (e.g. LiteLLM proxy without files_settings configured), falls back
    to reading the file as a base64-encoded PNG that can be sent inline.
    Returns (None, None) when no diagram path is provided.
    """
    if not file_path:
        return None, None
    try:
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="vision",
            )
        return result.id, None
    except Exception as exc:
        logging.warning(
            f"Could not upload diagram {file_path!r} to the vision API "
            f"({type(exc).__name__}: {exc}). Falling back to inline base64."
        )
    try:
        import base64
        with open(file_path, "rb") as f:
            return None, base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        logging.warning(
            f"Could not read diagram {file_path!r} for base64 encoding "
            f"({type(exc).__name__}: {exc}). Proceeding without diagram."
        )
        return None, None


with open(configuration.TEXT_INPUT_FILE) as f:
    _text_input = f.read()

_diagram_file_id, _diagram_b64 = get_vision_diagram(
    configuration.DIAGRAM_PATH, configuration.CLIENT)
system = SystemDescription(
    text_input=_text_input,
    file_id=_diagram_file_id,
    image_b64=_diagram_b64,
)

# saboteur = SaboteurHuman(configuration) OK
# saboteur = SaboteurLLMFaultTree(configuration) OK
# service_agent = ServiceAgentHuman(configuration) OK
# service_agent = ServiceAgentMock(configuration) OK
# service_agent = ServiceAgentLLM(configuration) OK
# assistant = DiagnosticAssistantMock(system, configuration) OK
# assistant = DiagnosticAssistantLLM(system, configuration) OK
# assistant = DiagnosticAssistantEvidenceKGOptimal(system, configuration) OK

match configuration.SABOTEUR_TYPE:
    case 'Human':
        saboteur = SaboteurHuman(configuration)
    case 'LLMFaultTree':
        saboteur = SaboteurLLMFaultTree(configuration)
    case 'FixedScenario':
        saboteur = SaboteurFixedScenario(configuration)
    case 'SpiceSim':
        saboteur = SaboteurSpiceSim(configuration)
    case _:
        raise ValueError(
            f'Unknown saboteur type: {configuration.SABOTEUR_TYPE}')
match configuration.SERVICE_TYPE:
    case 'Human':
        service_agent = ServiceAgentHuman(configuration)
    case 'LLM':
        service_agent = ServiceAgentLLM(configuration)
    case 'Mock':
        service_agent = ServiceAgentMock(configuration)
    case 'SpiceSim':
        service_agent = ServiceAgentSpiceSim(configuration)
    case 'SpiceSimMockNL':
        service_agent = ServiceAgentSpiceSimMockNL(configuration)
    case _:
        raise ValueError(
            f'Unknown service agent type: {configuration.SERVICE_TYPE}')
match configuration.ASSISTANT_TYPE:
    case 'EvidenceKGOptimal':
        assistant = DiagnosticAssistantEvidenceKGOptimal(system, configuration)
    case 'LLM':
        assistant = DiagnosticAssistantLLM(system, configuration)
    case 'Mock':
        assistant = DiagnosticAssistantMock(system, configuration)
    case 'RandomTrajectory':
        assistant = DiagnosticAssistantRandomTrajectory(system, configuration)
    case _:
        raise ValueError(
            f'Unknown assistant type: {configuration.ASSISTANT_TYPE}')

# saboteur = SaboteurHuman(configuration)
# service_agent = ServiceAgentHuman(configuration)
# assistant = DiagnosticAssistantEvidenceKGOptimal(system, configuration)
# # assistant = DiagnosticAssistantLLM(system, configuration)

scenario_logger = logging.getLogger("orchestrator")
scenario_logger.setLevel(configuration.LOG_LEVEL)
scenario_logger.propagate = False
scenario_logger.addHandler(configuration.get_file_handler())

chat_log = configuration.get_chat_log()
trajectory_log = configuration.get_trajectory_log(configuration.FORCED_SCENARIO_ID)

asyncio.run(run_diagnostic_scenario(system, saboteur,
            service_agent, assistant, scenario_logger, chat_log=chat_log,
            trajectory_log=trajectory_log))

# clear; python -m run_diagnostic_scenario --text-input-file /Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_description.txt --log-level 10 --rounds 5 --kg "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl" --system 3CubesSystem --ontology "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" --retrieval-folder "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Unstructured_knowledge_sources/3_cubes" --saboteur Human --service Human --assistant LLM

"""
clear; \
python -m run_diagnostic_scenario \
--text-input-file "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_description.txt" \
--diagram "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_schematics.png" \
--LLM-assistant-model "gpt-5.2" \
--log-level 10 \
--rounds 5 \
--kg "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/10_cubes/zorro-ontology-10-cubes-abox.ttl" \
--system 3CubesSystem \
--ontology "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
--retrieval-folder "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Unstructured_knowledge_sources/3_cubes" \
--saboteur Human \
--service Human \
--assistant EvidenceKGOptimal \
--interface cli
"""

# 10 cubes
"""
python -m run_diagnostic_scenario \
--text-input-file "Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_description.txt" \
--diagram "Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_schematics.png" \
--LLM-assistant-model "gpt-5.2" \
--NS-assistant-model "gpt-5.2" \
--log-level 10 \
--rounds 5 \
--kg "Knowledge_sources/Structured_knowledge_sources/10_cubes/zorro-ontology-10-cubes-abox.ttl" \
--system 10CubesSystem \
--ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
--retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/10_cubes" \
--saboteur Human \
--service Human \
--assistant EvidenceKGOptimal \
--interface cli \
--cache
"""

# Fixed scenario input
"""
python -m run_diagnostic_scenario \
--text-input-file "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_description.txt" \
--diagram "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_schematics.png" \
--LLM-assistant-model "gpt-5.2" \
--NS-assistant-model "gpt-5.2" \
--forced-scenario 0 \
--log-level 10 \
--rounds 5 \
--kg "Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl" \
--system 3CubesSystem \
--ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
--retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/3_cubes" \
--saboteur FixedScenario \
--service LLM \
--assistant LLM \
--interface cli 
"""

####################################################
####################################################
# go to for 3 cubes
"""
python -m run_diagnostic_scenario \
--text-input-file "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_description.txt" \
--diagram "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_schematics.png" \
--LLM-assistant-model "gpt-4.1" \
--NS-assistant-model "gpt-4.1" \
--log-level 10 \
--rounds 10 \
--system 3CubesSystem \
--kg "Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl" \
--ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
--retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/3_cubes" \
--saboteur Human \
--service Human \
--assistant LLM \
--interface cli 
"""

# go to for 10 cubes
"""
python -m run_diagnostic_scenario \
--text-input-file "Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_description.txt" \
--diagram "Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_schematics.png" \
--LLM-assistant-model "gpt-4.1" \
--NS-assistant-model "gpt-4.1" \
--log-level 10 \
--rounds 10 \
--system 10CubesSystem \
--kg "Knowledge_sources/Structured_knowledge_sources/10_cubes/zorro-ontology-10-cubes-abox.ttl" \
--ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
--retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/10_cubes" \
--saboteur Human \
--service Human \
--assistant LLM \
--interface cli 
"""


# go to for ambient light sensor
"""
python -m run_diagnostic_scenario \
--text-input-file "Knowledge_sources/Unstructured_knowledge_sources/ambient_light_sensor/ambient_light_sensor_description.txt" \
--diagram "Knowledge_sources/Unstructured_knowledge_sources/ambient_light_sensor/ambient_light_sensor_schematics.png" \
--LLM-assistant-model "gpt-4.1" \
--NS-assistant-model "gpt-4.1" \
--log-level 10 \
--rounds 10 \
--system AmbientLightSensorSystem \
--kg "Knowledge_sources/Structured_knowledge_sources/ambient_light_sensor/zorro-ontology-ambient-light-sensor-abox.ttl" \
--ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
--retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/ambient_light_sensor" \
--saboteur Human \
--service Human \
--assistant LLM \
--interface cli 
"""

####################################################

# 3 cubes LLM - voice
"""
python -m run_diagnostic_scenario \
--text-input-file "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_description.txt" \
--diagram "Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_schematics.png" \
--LLM-assistant-model "gpt-4.1" \
--NS-assistant-model "gpt-4.1" \
--log-level 10 \
--rounds 10 \
--system 3CubesSystem \
--kg "Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl" \
--ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
--retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/3_cubes" \
--saboteur Human \
--service Human \
--assistant LLM \
--interface voice 
"""
# 10 cubes symbolic - cli 
"""
python -m run_diagnostic_scenario \
--text-input-file "Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_description.txt" \
--diagram "Knowledge_sources/Unstructured_knowledge_sources/10_cubes/10_cubes_schematics.png" \
--LLM-assistant-model "gpt-4.1" \
--NS-assistant-model "gpt-4.1" \
--log-level 10 \
--rounds 10 \
--system 10CubesSystem \
--kg "Knowledge_sources/Structured_knowledge_sources/10_cubes/zorro-ontology-10-cubes-abox.ttl" \
--ontology "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" \
--retrieval-folder "Knowledge_sources/Unstructured_knowledge_sources/10_cubes" \
--saboteur Human \
--service Human \
--assistant EvidenceKGOptimal \
--interface cli 
"""

from datetime import datetime
import logging
import os
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Literal, Optional
from rdflib import URIRef, Namespace

from Utilities.chat_log import ChatLog


InterfaceType = Literal["cli", "voice"]
ZORRO = Namespace("http://www.example.org/zorro/")


@dataclass
class Configuration:
    SABOTEUR_TYPE: str
    SERVICE_TYPE: str
    ASSISTANT_TYPE: str

    TEXT_INPUT_FILE: str
    # OUTPUT_DIRECTORY: str = "Output"
    SYSTEM_NAME: str
    ONTOLOGY_PATH: Optional[str] = None
    ONTOLOGY_NAMESPACE: Optional[Namespace] = ZORRO
    KG_PATH: Optional[str] = None
    SYSTEM_URL: Optional[URIRef] = URIRef(base=ZORRO, value="System")
    DIAGRAM_PATH: Optional[str] = None
    FORCED_FAILURE_MODE: Optional[dict] = None

    RETRIEVAL_FOLDER_PATH: Optional[str] = None
    TOP_K: int = 3
    CHUNK_SIZE: int = 200
    CHUNK_OVERLAP: int = 20
    CACHE_PATH: str = "embeddings_cache.pkl"
    EMBED_MODEL: str = "text-embedding-3-small"
    TOKENIZER_MODEL: str = "cl100k_base"

    LLM_ASSISTANT_MODEL: str = "gpt-4.1"

    NS_ASSISTANT_MODEL: str = "gpt-4.1"

    SERVICE_MODEL: str = "gpt-4.1"

    # Per-agent config dicts — each agent reads its own keys with .get(key, default).
    # Passed via --assistant-config / --service-config / --saboteur-config JSON strings.
    # Values here take precedence over the individual model/RAG flags above.
    ASSISTANT_CONFIG: dict = field(default_factory=dict)
    SERVICE_CONFIG:   dict = field(default_factory=dict)
    SABOTEUR_CONFIG:  dict = field(default_factory=dict)

    FORCED_SCENARIO_ID: int = 0

    USE_CACHE: bool = False
    AGENTS_FORCED_TO_NO_CACHE: list[str] = field(default_factory=list)

    MAX_NUMBER_OF_ROUNDS: int = 10
    BATCH_RUN_SIZE: int = 10
    # if None it is left to the model, otherwise the diagnostic actions suggested will be a list of this length
    DIAGNOSTIC_ACTIONS_BATCH_SIZE: Optional[int] = 1

    HUMAN_TESTER: bool = False
    HUMAN_SYMPTOM_GENERATOR: bool = False
    INTERFACE_MODE: InterfaceType = "cli"
    # ensures that multiple instance will not share same client
    CLIENT: OpenAI = field(default_factory=OpenAI)

    LOG_PATH: str = "Logs/DebuggingLogs"
    CHAT_PATH: str = "Logs/Chats"
    TRAJECTORY_PATH: str = "Logs/Trajectories"
    LOG_FILE: str = field(
        default_factory=lambda: "DIAGNOSTIC_SCENARIO_RUN" + "_" + \
        datetime.now().isoformat(timespec='milliseconds')
    )
    LOG_LEVEL: int = 20

    def get_trajectory_log(self, scenario_id: int):
        from Utilities.trajectory_log import TrajectoryLog
        os.makedirs(self.TRAJECTORY_PATH, exist_ok=True)
        path = os.path.join(self.TRAJECTORY_PATH, self.LOG_FILE + "_TRAJECTORY.json")
        return TrajectoryLog(path, scenario_id)

    def get_chat_log(self) -> ChatLog:
        """Create and return a ChatLog writing to the same Logs directory."""
        os.makedirs(self.CHAT_PATH, exist_ok=True)
        path = os.path.join(self.CHAT_PATH,  self.LOG_FILE + "_CHAT.html")
        return ChatLog(path, session_info=self.LOG_FILE)

    def get_file_handler(self) -> logging.FileHandler:
        """Creates the file handler to be shared by loggers related to the diagnostic scenarios ran with this configuration"""
        
        from pathlib import Path

        self.LOG_PATH = Path(self.LOG_PATH)
        self.LOG_PATH.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            self.LOG_PATH / self.LOG_FILE,
            encoding="utf-8"
        )
        
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        file_handler.setFormatter(formatter)

        return file_handler

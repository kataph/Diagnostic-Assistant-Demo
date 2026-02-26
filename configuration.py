
from datetime import datetime
import logging
import os
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Literal, Optional
from rdflib import URIRef, Namespace



InterfaceType = Literal["cli", "voice"]
ZORRO = Namespace("http://www.example.org/zorro/")

@dataclass
class Configuration:
    SABOTEUR_TYPE: str
    SERVICE_TYPE: str
    ASSISTANT_TYPE: str
    
    TEXT_INPUT_FILE: str
    # OUTPUT_DIRECTORY: str = "Output"
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

    USE_CACHE: bool = False
    AGENTS_FORCED_TO_NO_CACHE: list[str] = field(default_factory=list)

    MAX_NUMBER_OF_ROUNDS: int = 10
    BATCH_RUN_SIZE: int = 10
    DIAGNOSTIC_ACTIONS_BATCH_SIZE: Optional[int] = 1 # if None it is left to the model, otherwise the diagnostic actions suggested will be a list of this length

    HUMAN_TESTER: bool = False
    HUMAN_SYMPTOM_GENERATOR: bool = False
    INTERFACE_MODE: InterfaceType = "cli"
    CLIENT: OpenAI = field(default_factory=OpenAI) # ensures that multiple instance will not share same client
    
    LOG_PATH: str = "Logs"
    LOG_FILE: str = "DIAGNOSTIC_SCENARIO_RUN" + "_" + datetime.now().isoformat(timespec='seconds')
    LOG_LEVEL: int = 20
    
    def get_file_handler(self) -> logging.FileHandler:
        """Creates the file handler to be shared by loggers related to the diagnostic scenarios ran with this configuration"""
        os.makedirs(self.LOG_PATH, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(self.LOG_PATH, self.LOG_FILE), encoding="utf-8")

        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        file_handler.setFormatter(formatter)
        
        return file_handler
    
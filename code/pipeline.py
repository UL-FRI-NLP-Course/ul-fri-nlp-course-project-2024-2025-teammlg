import enum
import logging
import os
import time
from typing import List, TypedDict

from model import Model


class ModelType(enum.StrEnum):
    DeepSeekSmall = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    DeepSeekBig = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    QwenSmall = "Qwen/Qwen3-4B"
    QwenBig = "Qwen/Qwen3-8B"


class RAGType(enum.Enum):
    NoRAG = enum.auto()
    SimpleRAG = enum.auto()
    AdvancedRAG = enum.auto()


class RetrievalSource(enum.Enum):
    TMDB = "tmdb"
    Letterboxd = "letterboxd"
    Wikipedia = "wiki"
    JustWatch = "justwatch"


class PipelineConfig(TypedDict):
    pipeline_name: str
    model_type: ModelType
    output_path: str
    rag_type: RAGType
    retrieval_sources: List[RetrievalSource]
    logging_level: int
    log_to_console: bool
    custom_prompt_template: str
    sample_response: bool


class Pipeline:
    def __init__(self, config: PipelineConfig):
        current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        self.pipeline_name = config.get("pipeline_name", f"pipeline_{current_timestamp}")
        self.output_path = config.get("output_path", None)
        self.rag_type = config.get("rag_type", RAGType.NoRAG)
        self.retrieval_sources = config.get("retrieval_sources", [
            RetrievalSource.JustWatch,
            RetrievalSource.Letterboxd,
            RetrievalSource.TMDB,
            RetrievalSource.Wikipedia
        ])
        self.custom_prompt_template = config.get("custom_prompt_template", None)
        self.sample_response = config.get("sample_response", False)

        if self.output_path is None:
            self.output_path = "pipeline_outputs"
        os.makedirs(self.output_path, exist_ok=True)

        self.logging_level = config.get("logging_level", logging.INFO)
        self.log_to_console = config.get("log_to_console", True)
        self.logger = logging.getLogger(f"pipeline")
        self.logger.setLevel(self.logging_level)
        self.logger.addHandler(logging.FileHandler(os.path.join(self.output_path, f"{self.pipeline_name}.log")))
        if self.log_to_console:
            self.logger.addHandler(logging.StreamHandler())

        self.model = Model(config["model_type"].value)
        if self.custom_prompt_template:
            self.model.set_chat_template(self.custom_prompt_template)

        self.logger.info(f"Pipeline {self.pipeline_name} initialized")

    def run_on_single_prompt(self, user_prompt: str) -> str:
        self.logger.info("Running pipeline...")
        self.logger.debug(f"Query: {user_prompt}")

        # TODO: Modify prompt
        prompt = [{
            "role": "user",
            "content": user_prompt
        }]

        # TODO: Extract data from prompt
        
        # TODO: Inject data into prompt

        self.logger.debug("Running model...")
        response = self.model.chat(prompt, sample=self.sample_response)

        self.logger.info("Pipeline run completed!")

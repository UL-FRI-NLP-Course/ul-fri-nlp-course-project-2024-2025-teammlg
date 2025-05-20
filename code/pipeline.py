import enum
import logging
import os
import time
from typing import List, Optional, TypedDict

from scraper_v2 import ScraperSource
from extraction import DocumentExtractionMethod, Extractor
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


class PipelineConfig(TypedDict):
    pipeline_name: str
    output_path: str
    model: Optional[Model]
    extractor: Optional[Extractor]
    scraping_sources: Optional[List[ScraperSource]]
    document_extraction_method: Optional[DocumentExtractionMethod]
    sample_response: bool


class Pipeline:
    def __init__(self, config: PipelineConfig):
        current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        self.pipeline_name = config.get("pipeline_name", f"pipeline_{current_timestamp}")
        self.output_path = config.get("output_path", None)

        if self.output_path is None:
            self.output_path = "pipeline_outputs"
        os.makedirs(self.output_path, exist_ok=True)

        logging_level = config.get("logging_level", logging.INFO)
        log_to_console = config.get("log_to_console", True)

        self.logger = logging.getLogger(f"pipeline")
        self.logger.setLevel(logging_level)
        self.logger.addHandler(logging.FileHandler(os.path.join(self.output_path, f"{self.pipeline_name}.log")))
        if log_to_console:
            self.logger.addHandler(logging.StreamHandler())

        self.model = config.get("model", None)
        self.extractor = config.get("extractor", None)
        self.scraping_sources = config.get("scraping_sources", [])
        self.document_extraction_method = config.get("document_extraction_method", DocumentExtractionMethod.Null)
        self.sample_response = config.get("sample_response")

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

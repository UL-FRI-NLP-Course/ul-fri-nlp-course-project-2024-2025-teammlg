from enum import StrEnum
from typing import Optional, TypedDict


class RAGType(StrEnum):
    Baseline = "baseline"
    Simple = "simple"
    Advanced = "advanced"


class ModelType(StrEnum):
    DeepSeek = "deepseek"
    Qwen = "qwen"


class PipelineConfig(TypedDict):
    model_type: ModelType  # Type of model (deepseek / qwen)
    rag_type: RAGType  # Type of RAG (baseline / simple / advanced)
    uses_memory: bool = False  # Should the model use memory (necessary for chatting)
    memory_capacity: int = 5  # How many prompts and replies should model keep in memory
    output_directory: Optional[str] = None  # Where should data be output


class PipelineOutput(TypedDict):
    user_prompt: str  # What user gave as an input
    data: str  # What was provided as additional info for LLM
    generated_response: str  # What LLM output
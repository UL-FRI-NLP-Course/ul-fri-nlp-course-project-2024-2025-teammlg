import enum
from typing import TypedDict


class ModelType(enum.Enum):
    DeepSeek = enum.auto()
    Qwen = enum.auto()


class ModelSize(enum.StrEnum):
    DeepSeekSmall = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    DeepSeekBig = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    QwenSmall = "Qwen/Qwen3-4B"
    QwenBig = "Qwen/Qwen3-8B"


class PipelineConfig(TypedDict):
    model_type: ModelType
    model_size: ModelSize
    output_name: str

"""
class Pipeline:
    def __init__(self, config: PipelineConfig):
        if config["model_type"] == ModelType.DeepSeek:
            self.model = 

    def run(user_prompt: str) -> str:
"""
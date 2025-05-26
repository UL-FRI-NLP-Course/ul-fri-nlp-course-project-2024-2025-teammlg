from enum import StrEnum
from typing import Optional, TypedDict

from rag import Rag
from qwen import QwenModel
from deepseek import DeepSeekModel

from aux import RAGType, ModelType, PipelineConfig, PipelineOutput


class ChatbotPipeline:
    def __init__(self, config: PipelineConfig):
        model_type = config["model_type"]
        rag_type = config["rag_type"]
        uses_memory = config.get("uses_memory", False)
        memory_capacity = config.get("memory_capacity", 5)
        output_directory = config.get("output_directory", None)
        
        self.rag_type = rag_type
        self.rag = Rag("rag_outputs")

        if model_type == ModelType.DeepSeek:
            if not output_directory:
                output_directory = "deepseek_model_outputs"
            self.model = DeepSeekModel(
                output_directory,
                uses_memory=uses_memory,
                memory_capacity=memory_capacity
            )
        elif model_type == ModelType.Qwen:
            if not output_directory:
                output_directory = "qwen_model_outputs"
            self.model = QwenModel(
                output_directory,
                uses_memory=uses_memory,
                memory_capacity=memory_capacity
            )
    
    def run(self, user_prompt: str) -> PipelineOutput:
        """Runs a single query through

        Args:
            user_prompt (str): The prompt from user

        Returns:
            PipelineOutput: The PipelineOutput object, containing response (see PipelineOutput docs)
        """
        is_baseline = self.rag_type == RAGType.Baseline
        data = ""

        if RAGType.Simple:
            documents = self.rag.get_simple_context(user_prompt)
            data = self.rag.data_to_str(documents)
        elif RAGType.Advanced:
            retrieval_tools = self.rag.get_retrieval_tools()
            instructions = self.model.answer_function_calling(user_prompt, retrieval_tools)
            documents = self.rag.get_context_from_tools(instructions)
            data = self.rag.data_to_str(documents)

        answer = self.model.answer_prompt(user_prompt, baseline=is_baseline, data=data)

        return {
            "user_prompt": user_prompt,
            "data": data,
            "generated_response": answer["assistant_response"]
        }
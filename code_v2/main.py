import argparse
from enum import StrEnum
import sys
from typing import Optional

from pipeline import ChatbotPipeline, ModelType, PipelineConfig, RAGType
import interactive_session
import auto_evaluation


class Operation(StrEnum):
    Evaluation = "evaluate"
    Conversation = "converse"


class CmdArguments(argparse.Namespace):
    rag_type: RAGType
    model_type: ModelType
    operation: Operation
    output_directory: Optional[str]
    evaluation_directory: str = "evaluation"
    uses_memory: bool = True
    memory_capacity: int = 5


if __name__ == "__main__":
    arguments = CmdArguments()
    parser = argparse.ArgumentParser()
    parser.parse_args(sys.argv, namespace=arguments)

    print("======== TeamMLG Movie Chatbot ========")

    config: PipelineConfig = {
        "rag_type": arguments.rag_type,
        "model_type": arguments.model_type,
        "output_directory": arguments.output_directory,
        "memory_capacity": arguments.memory_capacity,
        "uses_memory": arguments.uses_memory
    }
    pipeline = ChatbotPipeline(config)

    if arguments.operation == Operation.Conversation:
        interactive_session.converse(pipeline)
    elif arguments.operation == Operation.Evaluation:
        auto_evaluation.evaluate(pipeline, arguments.evaluation_directory)
    
    print("Exiting...")
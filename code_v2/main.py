import argparse
from enum import StrEnum
from typing import Optional

from aux import ModelType, PipelineConfig, RAGType


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
    parser = argparse.ArgumentParser(
        prog="TeamMLG Movie Chatbot",
        description="A simple chatbot for everything about films"
    )
    parser.add_argument("--rag_type", type=RAGType, required=True, help="What kind of RAG to perform", choices=list(RAGType))
    parser.add_argument("--model_type", type=ModelType, required=True, help="What LLM model to use", choices=list(ModelType))
    parser.add_argument("--operation", type=Operation, required=True, help="Whether to have an interactive session or do an automatic evaluation", choices=list(Operation))
    parser.add_argument("--output_directory", type=Optional[str], default=None, help="The directory to output conversations")
    parser.add_argument("--evaluation_directory", type=str, default="evaluation", help="The directory to output evaluation results")
    parser.add_argument("--uses_memory", type=bool, default=True, help="Whether the LLM should use chat history (important for chatting functionality)")
    parser.add_argument("--memory_capacity", type=int, default=5, help="How much of chat history to keep in memory")
    parser.parse_args(namespace=arguments)

    print("======== TeamMLG Movie Chatbot ========")

    # This is to avoid waiting 10 minutes only to find out
    # you messed up the initial parameters of the program
    from pipeline import ChatbotPipeline
    import interactive_session
    import auto_evaluation

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
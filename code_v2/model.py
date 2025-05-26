from abc import ABC, abstractmethod
from typing import List, TypedDict


class ChatElement(TypedDict):
    role: str
    content: str


class Model(ABC):
    def __init__(self, uses_memory: bool = False):
        self.uses_memory = uses_memory
        self.memory: List[ChatElement] = []

    def save_to_memory(self, content: str, role: str, to_back: bool = False):
        element = { "role": role, "content": content }
        if to_back:
            self.memory.insert(0, element)
        else:
            self.memory.append(element)

    @abstractmethod
    def answer_function_calling(self, prompt: str, tools: List) -> str:
        """The model determines which function should be called to answer the prompt correctly.

        Args:
            prompt (str): The prompt to answer
            tools (List): A list of functions that are available to call

        Returns:
            str: An answer containing function calls
        """
        pass

    @abstractmethod
    def answer_prompt(self, prompt: str) -> str:
        """The model answers the question in prompt.

        Args:
            prompt (str): The prompt to answer

        Returns:
            str: An answer
        """
        pass
from abc import ABC, abstractmethod
from typing import Any, Callable, List, TypedDict


class ChatElement(TypedDict):
    role: str
    content: str


class ModelAnswer(TypedDict):
    system_prompt: str
    """The system prompt that instructed the model how it should handle the query"""
    final_prompt: str
    """Final prompt, a combination of user prompt and retrieved data. This was provided as an input to the LLM"""
    assistant_response: str
    """The response that the LLM gave"""


class Model(ABC):
    def __init__(self, uses_memory: bool = False, memory_capacity: int = 5):
        self.uses_memory = uses_memory
        self.memory: List[ChatElement] = []
        self.memory_capacity = memory_capacity

    def save_to_memory(self, content: str, role: str):
        element = { "role": role, "content": content }
        self.memory.append(element)
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)
    
    def form_chat(self, system_prompt: str, final_user_prompt: str) -> List[ChatElement]:
        chat = self.memory.copy()
        chat.insert(0, { "role": "system", "content": system_prompt })
        chat.append({ "role": "user", "content": final_user_prompt })
        return chat

    @abstractmethod
    def answer_function_calling(self, prompt: str, tools: List[Callable]) -> str:
        """The model determines which function should be called to answer the prompt correctly.

        Args:
            prompt (str): The prompt to answer
            tools (List): A list of functions that are available to call

        Returns:
            str: An answer containing function calls
        """
        pass

    @abstractmethod
    def answer_prompt(self, prompt: str, data: Any = None, baseline: bool = True) -> ModelAnswer:
        """The model answers the question in prompt.

        Args:
            prompt (str): The prompt to answer
            data (Any): Data to incorporate into the prompt

        Returns:
            ModelAnswer: An answer object, see ModelAnswer docs
        """
        pass
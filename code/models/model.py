from abc import ABC, abstractmethod
from typing import Dict, Tuple

class Model(ABC):
    name: str
    requires_training: False

    # in case models have to be trained, fine-tuned, etc., if not just leave the function empty
    @abstractmethod
    def train(self, data):
        pass

    # return a reply to a query as string and the context
    @abstractmethod
    def reply(self, query) -> Tuple[str, Dict]:
        return "", {}

    # return your model's name
    def __str__(self):
        return f"Name: {self.name}"

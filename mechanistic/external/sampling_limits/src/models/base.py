from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs):
        pass

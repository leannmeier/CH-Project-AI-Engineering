from abc import ABC, abstractmethod
from schemas import ChatMessage

class BaseLLMClient(ABC):
    @abstractmethod
    async def generate_response(self, messages: list[ChatMessage]):
        pass
    
    @abstractmethod
    async def generate_response_stream(self, messages: list[ChatMessage]):
        pass
    

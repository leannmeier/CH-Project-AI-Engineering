from enum import Enum
from pydantic import BaseModel, computed_field
from Config import ClientType

class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    role: Role
    content: str
    
class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    
    @computed_field
    @property
    def total_tokens(self) -> int:
        """Calcula el total de tokens consumidos."""
        return self.input_tokens + self.output_tokens

class FinishReason(Enum):
    OK = 'ok'
    TOKEN_LIMIT = 'token_limit'
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    STOP = 'stop'
    
class ModelResponse(BaseModel):
    content: str # contenido de la respuesta
    provider: ClientType # proveedor que realizo la respuesta
    model: str # modelo exacto que realizo la respuesta
    usage: Usage # esto tendra información sobre el consumo 
    finish_reason: FinishReason # motivo de la finalizacion de la respuesta
    
from enum import Enum
from pydantic import BaseModel, computed_field, Field
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
    content: str 
    provider: ClientType 
    model: str 
    usage: Usage  
    finish_reason: FinishReason 
    
class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"

class AnalisisTecnico(BaseModel):
    tecnologias: list[str] = Field(
        description="Lista de tecnologia, bases de datos, frameworks o herramientas identificadas en el texto"
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Nivel de criticidad evaluado a partir del impacto tecnico descrito (baja, media o alta)"
    )
    resumen_tecnico: str = Field(
        description="Resumen tecnico breve, claro y directo del problema o escenario analizado."
    )
    
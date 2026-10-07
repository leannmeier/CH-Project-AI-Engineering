from enum import Enum
from pydantic import BaseModel, SecretStr, Field

class ClientType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = 'groq' # Como no dispongo de saldo en las cuentas de OpenAI y Anthropic, uso GROQ ya que tiene un free tier generoso para estas pruebas

class Config(BaseModel):
    provider: ClientType
    model: str
    api_key: SecretStr
    temperature: float = Field(default= 0.7, ge=0.0, le=2.0,description="Temperatura de la respuesta generada por el modelo, entre 0 y 2.") 
    max_tokens: int = Field(default=1024 , ge=1, description="Límite máximo de tokens a generar por la respuesta")
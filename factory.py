import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, SecretStr

from Config import Config, ClientType

load_dotenv()

class LLMFactory:
    
    @staticmethod
    def create_llm(config: Config) -> BaseChatModel:
        api_key_str = config.api_key.get_secret_value()
        
        if config.provider == ClientType.OPENAI:
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                api_key=SecretStr(api_key_str)
            )
        elif config.provider == ClientType.ANTHROPIC:
            return ChatAnthropic(
                model_name=config.model,
                temperature=config.temperature,
                api_key=SecretStr(api_key_str)
            )
        else:
            raise ValueError(f"Proveedor no soportado: {config.provider}")

    @staticmethod
    def get_default_config(provider: ClientType = ClientType.OPENAI) -> Config:
        """Helper para construir un Config leyendo las API Keys reales del entorno."""
        if provider == ClientType.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            model = "gpt-4o-mini"
        elif provider == ClientType.ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            model = "claude-3-5-sonnet-20241022"
        else:
            raise ValueError(f"Proveedor no válido: {provider}")

        return Config(
            provider=provider,
            model=model,
            api_key=SecretStr(api_key),
            temperature=0.0
        )
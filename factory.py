import os
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

from Config import Config, ClientType

load_dotenv()

class LLMFactory:
    """Factory centralizado para crear instancias de LLM y Embeddings."""

    @staticmethod
    def create_llm(config: Config) -> BaseChatModel:
        api_key_str = config.api_key.get_secret_value()
        
        if config.provider == ClientType.OPENAI:
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                api_key=api_key_str
            )
        elif config.provider == ClientType.ANTHROPIC:
            return ChatAnthropic(
                model=config.model,
                temperature=config.temperature,
                api_key=api_key_str
            )
        else:
            raise ValueError(f"Proveedor no soportado para LLM: {config.provider}")

    @staticmethod
    def create_embeddings(config: Optional[Config] = None) -> Embeddings:
        """
        Retorna un modelo de embeddings local gratuito (all-MiniLM-L6-v2) 
        para evitar errores de falta de cuota en APIs externas.
        """
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    @staticmethod
    def get_default_config(provider: ClientType = ClientType.OPENAI) -> Config:
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
            api_key=api_key,
            temperature=0.0
        )
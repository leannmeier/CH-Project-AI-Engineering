import asyncio
import os
from dotenv import load_dotenv
from pydantic import SecretStr

from BaseLLMClient import BaseLLMClient
from Config import Config, ClientType
from exceptions import LLMClientError
from models.AnthropicClient import AnthropicClient
from models.OpenAIClient import OpenAIClient
from schemas import ChatMessage, Role

load_dotenv()

async def probar_cliente(client: BaseLLMClient, nombre_proveedor: str, pregunta: str):
    """
    Función auxiliar genérica para probar cualquier cliente que herede de BaseLLMClient.
    Aplica el principio de intercambiabilidad (polimorfismo).
    """
    print(f"\n==================================================")
    print(f"   PROBANDO CLIENTE: {nombre_proveedor.upper()}")
    print(f"==================================================")

    mensajes = [ChatMessage(role=Role.USER, content=pregunta)]

    # 1. Prueba en modo Normal
    print("\n--- 1. Respuesta Completa (generate_response) ---")
    try:
        respuesta = await client.generate_response(mensajes)
        print(f"Modelo utilizado: {respuesta.model}")
        print(f"Tokens usados: Input={respuesta.usage.input_tokens}, Output={respuesta.usage.output_tokens}")
        print(f"Contenido:\n{respuesta.content}")
    except LLMClientError as e:
        print(f"Error al generar respuesta: {e}")

    # 2. Prueba en modo Streaming
    print("\n--- 2. Respuesta en Streaming (generate_response_stream) ---")
    try:
        print("Respuesta: ", end="", flush=True)
        async for chunk in client.generate_response_stream(mensajes):
            print(chunk, end="", flush=True)
        print("\n\n¡Streaming completado con éxito!")
    except LLMClientError as e:
        print(f"\nError durante el streaming: {e}")


async def main():
    pregunta = "¿Qué es la entropía?"
    # Instanciación y prueba del cliente de Anthropic
    # ------------------------------------------------------------------
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        config_anthropic = Config(
            provider=ClientType.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            api_key=SecretStr(anthropic_api_key),
            temperature=0.7,
            max_tokens=1024,
        )
        anthropic_client = AnthropicClient(config_anthropic)
        await probar_cliente(anthropic_client, "Anthropic", pregunta)
    else:
        print("\nNo se encontró ANTHROPIC_API_KEY en el entorno. Se omite la prueba de Anthropic.")
        
    # Instanciación y prueba del cliente de OpenAI
    # ------------------------------------------------------------------
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        config_openai = Config(
            provider=ClientType.OPENAI,
            model="gpt-4o-mini",
            api_key=SecretStr(openai_api_key),
            temperature=0.7,
            max_tokens=1024,
        )
        openai_client = OpenAIClient(config_openai)
        await probar_cliente(openai_client, "OpenAI", pregunta)
    else:
        print("\nNo se encontró OPENAI_API_KEY en el entorno. Se omite la prueba de OpenAI.")


if __name__ == "__main__":
    asyncio.run(main())
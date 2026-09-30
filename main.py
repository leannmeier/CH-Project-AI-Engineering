import asyncio
import json
from chain import process_text
from factory import LLMFactory
from Config import ClientType


async def main():
    texto_prueba = """
    [ERROR CRÍTICO] 2026-03-29 14:32:10 - Fallo de conexión en clúster de MongoDB Atlas.
    El servicio principal corriendo en Docker superó el tiempo de espera en las consultas.
    La capa de caché en Redis no responde. FastAPI devuelve 500 en producción.
    """

    print("--- 1. Ejecución por defecto (OpenAI desde variables de entorno) ---")
    try:
        resultado = await process_text(texto_prueba)
        print(json.dumps(resultado.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- 2. Ejemplo de ejecución pasando configuración personalizada ---")
    try:
        config_anthropic = LLMFactory.get_default_config(ClientType.ANTHROPIC)
        resultado_anthropic = await process_text(texto_prueba, config=config_anthropic)
        print(json.dumps(resultado_anthropic.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
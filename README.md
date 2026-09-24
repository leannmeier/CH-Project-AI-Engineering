# Sistema Intelligence — Unified Async LLM Client

Cliente asíncrono unificado para interactuar con múltiples proveedores de LLM (Anthropic y OpenAI) bajo una única interfaz común, con validación de datos vía Pydantic y soporte de streaming.

## Arquitectura

- **`BaseLLMClient`** — clase abstracta (ABC) que define el contrato común: `generate_response` (respuesta completa) y `generate_response_stream` (generador asíncrono de tokens).
- **`AnthropicClient`** / **`OpenAIClient`** (en `models/`) — implementaciones concretas del contrato, cada una traduciendo el formato específico de su SDK (mensajes, motivo de finalización, uso de tokens) a los esquemas propios del proyecto.
- **`schemas.py`** — modelos Pydantic: `ChatMessage` (mensajes de entrada), `ModelResponse` y `Usage` (respuesta normalizada, independiente del proveedor), `FinishReason` (motivo de finalización unificado).
- **`Config.py`** — configuración validada con Pydantic (proveedor, modelo, API key protegida con `SecretStr`, temperatura, máximo de tokens).
- **`exceptions.py`** — `LLMClientError`, excepción propia que envuelve los errores específicos de cada SDK (`anthropic.APIError`, `openai.APIError`) bajo un solo tipo genérico.
- **`main.py`** — script de validación: prueba ambos clientes (si hay API key configurada para cada uno) en modo normal y en modo streaming.
## Requisitos

- Python 3.12
- Una cuenta con API key de Anthropic y/o OpenAI (con crédito cargado para poder hacer llamadas reales)
## Instalación

```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/Scripts/activate   # Git Bash / Linux / Mac
# .venv\Scripts\activate.bat    # cmd de Windows

# Instalar dependencias
pip install -r requirements.txt
```
## Configuración

Copiá `.env.example` a `.env` y completá tus API keys reales:

```bash
cp .env.example .env
```
## Autor
Meier Leandro Agustin - Analista de Sistemas
# Sistema Intelligence — Unified Async LLM Client

Cliente asíncrono unificado para interactuar con múltiples proveedores de LLM (Anthropic y OpenAI) bajo una única interfaz común, con validación de datos vía Pydantic, soporte de streaming, y un pipeline LCEL con salida estructurada y resiliente.

## Arquitectura

### Módulo 1 — Cliente async unificado
- **`BaseLLMClient`** — clase abstracta (ABC) que define el contrato común: `generate_response` (respuesta completa) y `generate_response_stream` (generador asíncrono de tokens).
- **`AnthropicClient`** / **`OpenAIClient`** (en `models/`) — implementaciones concretas del contrato, cada una traduciendo el formato específico de su SDK (mensajes, motivo de finalización, uso de tokens) a los esquemas propios del proyecto.
- **`schemas.py`** — modelos Pydantic: `ChatMessage`, `ModelResponse`/`Usage` (respuesta normalizada), `FinishReason`, y `AnalisisTecnico` (esquema de salida estructurada del Módulo 2).
- **`Config.py`** — configuración validada con Pydantic (proveedor, modelo, API key protegida con `SecretStr`, temperatura 0-2, `max_tokens` ≥ 1).
- **`exceptions.py`** — `LLMClientError`, excepción propia que envuelve los errores específicos de cada SDK bajo un solo tipo genérico.
- **`factory.py`** — `LLMFactory`, resuelve el cliente LangChain (`ChatOpenAI`/`ChatAnthropic`) correcto a partir de `Config.provider`, sin que el código que lo usa tenga que conocer la lógica de selección.
- **`main.py`** — script de validación del Módulo 1: prueba ambos clientes en modo normal y streaming.

### Módulo 2 — Pipeline LCEL con salida estructurada
- **`chain.py`** — pipeline construido con LangChain Expression Language (`prompt | llm.with_structured_output(...)`), que analiza un texto libre y devuelve un objeto `AnalisisTecnico` validado (tecnologías detectadas, nivel de criticidad, resumen técnico). Incluye reintentos automáticos con backoff exponencial (`with_retry`) ante fallos transitorios de la API.
- **`main.py`** (Módulo 2) — ejecuta el pipeline con la configuración por defecto (OpenAI) y con una configuración personalizada (Anthropic), imprimiendo el resultado validado como JSON.

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
```
ANTHROPIC_API_KEY=tu-api-key-real-de-anthropic
OPENAI_API_KEY=tu-api-key-real-de-openai
```

## Uso

**Módulo 1** (cliente async unificado, respuesta normal y streaming):
```bash
python main.py
```

**Módulo 2** (pipeline LCEL de análisis técnico estructurado):
```bash
python -m chain  # o el punto de entrada que corresponda a la Pre-entrega 2
```

## Notas de implementación

- **Anthropic (SDK ≥ 1.0):** el SDK crudo (`anthropic`) eliminó `temperature` de `messages.create()`; el `AnthropicClient` del Módulo 1 no lo reenvía por este motivo. El wrapper de LangChain (`langchain_anthropic.ChatAnthropic`), en cambio, sí acepta `temperature` y lo maneja internamente sin ese conflicto.
- **Manejo de errores:** ambos clientes del Módulo 1 capturan la excepción base de su SDK y la traducen a `LLMClientError`, agnóstica de proveedor.
- **Resiliencia (Módulo 2):** el pipeline LCEL reintenta automáticamente (hasta 3 intentos, con backoff exponencial) ante errores transitorios antes de propagar la excepción.

## Autor
Meier Leandro Agustin - Analista de Sistemas
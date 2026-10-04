# Sistema Intelligence — Unified Async LLM Client

Cliente asíncrono unificado para interactuar con múltiples proveedores de LLM (Anthropic y OpenAI) bajo una única interfaz común, con validación de datos vía Pydantic, soporte de streaming, pipelines LCEL con salida estructurada/resiliente, y un sistema RAG local sobre ChromaDB.

## Arquitectura

### Módulo 1 — Cliente async unificado
- **`BaseLLMClient`** — clase abstracta (ABC) que define el contrato común: `generate_response` (respuesta completa) y `generate_response_stream` (generador asíncrono de tokens).
- **`AnthropicClient`** / **`OpenAIClient`** (en `models/`) — implementaciones concretas del contrato, cada una traduciendo el formato específico de su SDK a los esquemas propios del proyecto.
- **`schemas.py`** — modelos Pydantic del proyecto (`ChatMessage`, `ModelResponse`/`Usage`, `FinishReason`, `AnalisisTecnico`).
- **`Config.py`** — configuración validada con Pydantic (proveedor, modelo, API key protegida con `SecretStr`, temperatura 0-2, `max_tokens` ≥ 1).
- **`exceptions.py`** — `LLMClientError`, excepción propia que envuelve los errores específicos de cada SDK bajo un solo tipo genérico.
- **`factory.py`** — `LLMFactory`, resuelve el cliente LLM y el modelo de embeddings correctos a partir de `Config.provider`.

### Módulo 2 — Pipeline LCEL con salida estructurada
- **`chain.py`** — pipeline (`prompt | llm.with_structured_output(AnalisisTecnico)`) que analiza un texto libre y devuelve un objeto validado (tecnologías, criticidad, resumen). Incluye reintentos automáticos con backoff exponencial (`with_retry`) ante fallos transitorios.

### Módulo 3 — Sistema RAG local con ChromaDB
- **`ingest.py`** — carga documentos (`.pdf`/`.txt`) desde `./data` y los fragmenta con `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=150), preservando metadata de origen.
- **`vectorstore.py`** — indexa los chunks en ChromaDB local (persistido en `./chroma_db`), usando embeddings locales (`HuggingFaceEmbeddings`, `all-MiniLM-L6-v2`) para no depender de crédito de API. Evita reindexar si ya hay datos cargados.
- **`rag_chain.py`** — pipeline LCEL de RAG: recupera los `k=3` chunks más relevantes vía `RunnableParallel` (preservando los `Document` originales para trazabilidad), genera la respuesta usando solo ese contexto, y valida tanto la entrada (`RAGQueryInput`) como la salida (`RAGQueryOutput`, con `sources: list[DocumentSource]` reflejando los fragmentos realmente usados). El prompt instruye al modelo a admitir cuando no tiene información suficiente, en vez de inventar.
- **`main.py`** — **punto de entrada actual del proyecto**: indexa `./data` (si hace falta) y ejecuta una consulta de prueba sobre el pipeline RAG, mostrando la respuesta y sus fuentes.

> Nota: `main.py` fue evolucionando con cada módulo y hoy ejecuta el flujo del Módulo 3 (RAG). Los módulos 1 y 2 siguen disponibles como componentes (`models/`, `chain.py`) e instanciables por separado, pero no tienen un script de entrada propio activo en este momento.

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
ANTHROPIC_API_KEY=tu-api-key-real-de-anthropic
OPENAI_API_KEY=tu-api-key-real-de-openai

## Uso

1. Colocá tus documentos (`.pdf` o `.txt`) en la carpeta `./data`.
2. Corré:
```bash
python main.py
```
La primera ejecución indexa los documentos; las siguientes reutilizan el índice ya creado en `./chroma_db`.

## Notas de implementación

- **Anthropic (SDK ≥ 1.0):** el SDK crudo (`anthropic`) eliminó `temperature` de `messages.create()`; el `AnthropicClient` del Módulo 1 no lo reenvía por este motivo. El wrapper de LangChain (`langchain_anthropic.ChatAnthropic`) sí lo acepta y lo maneja internamente sin ese conflicto.
- **Manejo de errores:** ambos clientes del Módulo 1 capturan la excepción base de su SDK y la traducen a `LLMClientError`, agnóstica de proveedor.
- **Resiliencia (Módulo 2):** el pipeline LCEL reintenta automáticamente (hasta 3 intentos, con backoff exponencial) ante errores transitorios.
- **Embeddings locales (Módulo 3):** se usa `HuggingFaceEmbeddings` en vez de embeddings de OpenAI/Anthropic para que la indexación funcione sin consumir crédito de API.

## Autor
Meier Leandro Agustín — Analista de Sistemas
# Sistema Intelligence — Unified Async LLM Client

Cliente asíncrono unificado para interactuar con múltiples proveedores de LLM (Anthropic y OpenAI) bajo una única interfaz común, con validación de datos vía Pydantic, soporte de streaming, pipelines LCEL con salida estructurada/resiliente, y un sistema RAG avanzado escalable en Pinecone con recuperación híbrida.

## Arquitectura

### Módulo 1 — Cliente async unificado
- **`BaseLLMClient`** — clase abstracta (ABC) que define el contrato común: `generate_response` (respuesta completa) y `generate_response_stream` (generador asíncrono de tokens).
- **`AnthropicClient`** / **`OpenAIClient`** (en `models/`) — implementaciones concretas del contrato, cada una traduciendo el formato específico de su SDK a los esquemas propios del proyecto.
- **`schemas.py`** — modelos Pydantic del proyecto (`ChatMessage`, `ModelResponse`/`Usage`, `FinishReason`, `AnalisisTecnico`).
- **`Config.py`** — configuración validada con Pydantic (proveedor, modelo, API key protegida con `SecretStr`, temperatura 0-2, `max_tokens` ≥ 1).
- **`exceptions.py`** — `LLMClientError`, excepción propia que envuelve los errores específicos de cada SDK bajo un solo tipo genérico.
- **`factory.py`** — `LLMFactory`, resuelve el cliente LLM (`ChatOpenAI`/`ChatAnthropic`) y el modelo de embeddings correctos a partir de `Config.provider`.

### Módulo 2 — Pipeline LCEL con salida estructurada
- **`chain.py`** — pipeline (`prompt | llm.with_structured_output(AnalisisTecnico)`) que analiza un texto libre y devuelve un objeto validado (tecnologías, criticidad, resumen). Incluye reintentos automáticos con backoff exponencial (`with_retry`) ante fallos transitorios.

### Módulo 3 — RAG local con ChromaDB
- **`ingest.py`** — carga documentos (`.pdf`/`.txt`, UTF-8) desde `./data` y los fragmenta con `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=150), preservando metadata de origen.
- Pipeline RAG local inicial sobre ChromaDB, luego migrado a Pinecone en el Módulo 4 (ver abajo).

### Módulo 4 — RAG avanzado y escalable con Pinecone
- **`vectorstore.py`** — migrado de ChromaDB local a **Pinecone Serverless** (`PineconeVectorStore`). Crea el índice automáticamente si no existe, con dimensión ajustada al modelo de embeddings local (`all-MiniLM-L6-v2`, 384 dimensiones) y métrica coseno. Sigue usando embeddings locales de HuggingFace para no depender de crédito de API.
- **`rag_chain.py`** — `build_hybrid_retriever` combina un retriever **vectorial** (Pinecone) con un retriever **léxico** (`BM25Retriever`, local) mediante `EnsembleRetriever` (ponderación 60% vectorial / 40% léxico), para cubrir tanto similitud semántica como coincidencias exactas de términos.
- **`evaluate.py`** — script de evaluación cuantitativa: ejecuta el recuperador híbrido contra un "Golden Set" de preguntas con una palabra clave esperada, y calcula **Precision@5** y **Recall@5** por pregunta y en promedio. No depende de ningún LLM (solo recuperación), así que corre sin necesidad de crédito de OpenAI/Anthropic.
- **`main.py`** — con el flag `--evaluate` corre `evaluate.py` en lugar del flujo normal de consulta RAG.

## Requisitos

- Python 3.12
- Una cuenta con API key de Anthropic y/o OpenAI (con crédito cargado para poder hacer llamadas reales de generación)
- Una cuenta de Pinecone (free tier) con su API key

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

Copiá `.env.example` a `.env` y completá tus credenciales reales:

```bash
cp .env.example .env
```
ANTHROPIC_API_KEY=tu-api-key-real-de-anthropic
OPENAI_API_KEY=tu-api-key-real-de-openai
PINECONE_API_KEY=tu-api-key-real-de-pinecone
INDEX_NAME=rag-hybrid-index

## Uso

1. Colocá tus documentos (`.pdf` o `.txt`, codificados en UTF-8) en la carpeta `./data`.
2. Ejecutar el flujo completo (indexación + consulta RAG híbrida con generación de respuesta):
```bash
python main.py
```
3. Ejecutar solo la evaluación cuantitativa del recuperador (no requiere crédito de LLM):
```bash
python main.py --evaluate
```

La primera ejecución crea el índice en Pinecone e indexa los documentos; las siguientes reutilizan el índice existente. Si cambiás el contenido de `./data`, hace falta reindexar (cambiando `INDEX_NAME` a uno nuevo, o vaciando el índice existente desde el dashboard de Pinecone).

## Notas de implementación

- **Anthropic (SDK ≥ 1.0):** el SDK crudo (`anthropic`) eliminó `temperature` de `messages.create()`; el `AnthropicClient` del Módulo 1 no lo reenvía por este motivo. El wrapper de LangChain (`langchain_anthropic.ChatAnthropic`) sí lo acepta y lo maneja internamente sin ese conflicto.
- **Manejo de errores:** ambos clientes del Módulo 1 capturan la excepción base de su SDK y la traducen a `LLMClientError`, agnóstica de proveedor.
- **Resiliencia (Módulo 2):** el pipeline LCEL reintenta automáticamente (hasta 3 intentos, con backoff exponencial) ante errores transitorios.
- **Embeddings locales:** se usa `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`, 384 dimensiones) en lugar de embeddings de OpenAI/Anthropic, para que la indexación y recuperación funcionen sin consumir crédito de API — solo la generación final de texto depende del proveedor LLM configurado.
- **LangChain ≥ 1.0:** la reorganización del paquete movió componentes "clásicos" (como `EnsembleRetriever`) a `langchain-classic`, separado del núcleo `langchain` (enfocado ahora en agentes). Ver `requirements.txt` para las dependencias exactas instaladas.
- **Dimensión del índice de Pinecone:** debe coincidir con la dimensión del modelo de embeddings usado (384 para `all-MiniLM-L6-v2`), no con la de modelos de OpenAI (1536) si no se usan esos embeddings.

## Autor
Meier Leandro Agustín — Analista de Sistemas
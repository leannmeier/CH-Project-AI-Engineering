# Entrega final — Sistema Intelligence

Esta entrega integra en **un único repositorio** los módulos previos: abstracción
de LLMs, recuperación híbrida con Pinecone, Supervisor y agentes LangGraph,
API FastAPI, checkpoints Redis, aprobación humana y observabilidad Phoenix.

## Arquitectura

```mermaid
flowchart LR
    U[Cliente] --> API[FastAPI]
    API -->|POST /tasks| Q[Job en Redis]
    Q --> G[Supervisor LangGraph]
    G --> R[Investigación con Tavily]
    G --> A[Análisis y métricas]
    R --> G
    A --> G
    G --> H{Aprobación humana}
    H -->|Aprobar| P[Publicar informe en Redis]
    H -->|Rechazar| X[No publicar]
    G --> C[(Checkpointer Redis)]
    G -.-> O[Phoenix trazas]
    API -->|POST /knowledge/search| K[RAGSystem híbrido]
    K --> B[BM25 local]
    K --> V[(Pinecone Serverless)]
    ```

**Alcance de la integración:** el RAG está disponible desde la misma API
mediante `/knowledge/search`. El Supervisor de los módulos 6–7 conserva
su herramienta de investigación web y no consume automáticamente RAG;
la selección dinámica entre conocimiento local y web sería una ampliación
posterior, no una funcionalidad ya implementada.

## Requisitos para ejecutar

- Python 3.12 o posterior; Git Bash en Windows (o bash en Linux/macOS).
- Docker con Compose habilitado.
- Entorno virtual con `pip install -r requirements.txt`.
- Copiar `.env.example` a `.env` y configurar `GROQ_API_KEY` y
  `TAVILY_API_KEY` para usar agentes reales.
- Para el endpoint RAG, `PINECONE_API_KEY`, `INDEX_NAME` y un índice
  ya cargado de dimensión 384 con embeddings `all-MiniLM-L6-v2`.
- El archivo `.env` no debe incluirse en el repositorio ni mostrarse en capturas.

## Iniciar localmente con una orden

Tras preparar Docker, Python y `.env`, ejecutar desde la raíz:

```bash
bash run.sh
```

El script levanta Redis Stack con Docker Compose, espera su estado
saludable e inicia FastAPI en http://127.0.0.1:8000/docs.
No inicializa Pinecone ni elimina datos existentes.

Opcionalmente, activar Phoenix en otra consola **antes** de iniciar la API:

```bash
docker compose --profile observability up -d
```

La interfaz de Phoenix se encuentra en http://localhost:6006.
Asegurarse de que `ENABLE_TRACING=1` en `.env`.

## Endpoints

| Método | Ruta | Función |
|---|---|---|
| GET | `/health` | Estado de Redis |
| POST | `/tasks` | Encola un trabajo de agentes |
| GET | `/tasks/{job_id}` | Consulta el estado del trabajo |
| POST | `/tasks/{job_id}/approve` | Aprueba o rechaza el reporte |
| POST | `/knowledge/search` | Recupera hasta cinco fragmentos con `source` y metadatos |

Ejemplo de recuperación documental:

```json
{
  "question": "¿Cómo leer texto UTF-8 con pathlib?",
  "k": 5
}
```

Enviarlo a `POST /knowledge/search` desde Swagger (`/docs`).
La respuesta tiene `question` y `hits`, cada hit contiene
`source`, `content` y `metadata`.

La API no debe revelar claves API y devuelve 503 si falla la recuperación
externa. Los agentes pueden tener otros fallos (créditos, límites o
conectividad), registrados como trabajos FAILED.

## Comprobación de calidad antes de entregar

Ejecutar desde la raíz:

```bash
python -m pytest tests/test_rag_evaluation.py tests/test_api.py tests/test_knowledge_api.py -v
```

Las pruebas unitarias usan dobles de Pinecone, Redis y agentes. Se deben
complementar con verificación manual de los servicios reales:

1. Levantar Redis Stack y la API, comprobar `GET /health`.
2. Crear un trabajo, consultar su estado hasta `WAITING_APPROVAL`,
   aprobarlo y comprobar `DONE` / `published=true`.
3. Repetir con rechazo, comprobar `REJECTED` / `published=false`.
4. Enviar cinco solicitudes concurrentes (ver `scripts/load_test.py`).
5. Capturar al menos cinco trazas reales en Phoenix y adjuntar evidencia
   en `screenshots/` antes de entregar.
6. Verificar `POST /knowledge/search` con el índice de pruebas de Pinecone.
7. Ejecutar `python evaluate.py` y guardar resultados reproducibles.
8. Instalar dependencias en un entorno limpio con Python 3.12+.

**Estado de evidencia:** para el módulo 4 se reportaron 17 pruebas aprobadas,
ingesta de 11 chunks y un Golden Set con Recall@5 de 100% y Precision@5
de 36% sobre un índice de pruebas. Esto no reemplaza la ejecución actual
de las pruebas de la entrega final ni acredita por sí solo trazas reales
de Phoenix.

## Decisiones técnicas y límites

- Se usa `async/await` en FastAPI y agentes; el inicializador del
  recuperador RAG (que carga embeddings) se ejecuta en `asyncio.to_thread`
  para evitar bloquear el event loop durante su construcción.
- Pydantic valida la entrada de `/tasks` y `/knowledge/search` y las
  respuestas de consulta y persistencia.
- Redis Stack mantiene jobs y checkpoints. El proceso Python ejecuta
  workers locales con `asyncio.create_task`; **no** es una cola distribuida
  duradera y los trabajos en ejecución pueden interrumpirse si reinicia el
  servidor. La persistencia de checkpoints no implica reanudación
  automática de todos los trabajos pendientes.
- La publicación de informes en Redis es una acción crítica simulada.
- Un índice Pinecone nuevo permite reindexar sin alterar el índice previo.
- El recuperador híbrido y sus metadatos se evalúan por separado de la
  calidad semántica final del texto del agente.
- Un despliegue local reproducible es suficiente para la consigna; no
  se requiere CI/CD ni hosting de producción.

"""API de producción del sistema multi-agente.

Endpoints:
  POST /tasks                  encola una consulta y devuelve el job_id al instante (202)
  GET  /tasks/{id}             estado del job (PENDING, RUNNING, WAITING_APPROVAL, DONE, REJECTED, FAILED)
  POST /tasks/{id}/approve     decisión humana sobre la acción crítica (publicar el reporte)
  GET  /health                 verifica la conexión con Redis

Arrancar desde la raíz del repo:  uvicorn app.main:app --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from app.observability import setup_observability  # noqa: E402

setup_observability()  # antes de importar/crear el grafo y los modelos

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.graph import build_graph, open_checkpointer  # noqa: E402
from app.jobs import REDIS_URL, Job, JobStatus, JobStore  # noqa: E402
from app.worker import resume_job, run_job  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("app.main")


class TaskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500, description="Consulta para el sistema multi-agente.")


class ApprovalRequest(BaseModel):
    approved: bool
    comment: Optional[str] = None


def _spawn(app: FastAPI, coro) -> None:
    """Lanza la tarea en segundo plano y guarda una referencia para que no la recoja el garbage collector."""
    task = asyncio.create_task(coro)
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = JobStore.from_url()
    try:
        await store.ping()  # falla rápido y con un mensaje claro si Redis no está levantado
    except Exception:
        logger.error("No se pudo conectar a Redis en %s. ¿Levantaste `docker compose up -d`?", REDIS_URL)
        await store.close()
        raise

    try:
        async with open_checkpointer() as checkpointer:
            app.state.store = store
            app.state.graph = build_graph(checkpointer)
            app.state.tasks = set()
            logger.info("API lista (Redis OK, grafo compilado).")
            yield
    finally:
        await store.close()


app = FastAPI(title="Sistema Intelligence - API multi-agente", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    try:
        await request.app.state.store.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis no responde: {exc}")
    return {"status": "ok"}


@app.post("/tasks", status_code=202)
async def create_task(body: TaskRequest, request: Request):
    store: JobStore = request.app.state.store
    job = await store.create(body.query)
    _spawn(request.app, run_job(request.app.state.graph, store, job.id, body.query))
    return {"job_id": job.id, "status": job.status}


@app.get("/tasks/{job_id}", response_model=Job)
async def get_task(job_id: str, request: Request):
    job = await request.app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado (o expirado).")
    return job


@app.post("/tasks/{job_id}/approve", status_code=202)
async def approve_task(job_id: str, body: ApprovalRequest, request: Request):
    store: JobStore = request.app.state.store
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado (o expirado).")
    if job.status != JobStatus.WAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"El job está en estado {job.status.value}; solo se aprueba uno en WAITING_APPROVAL.",
        )

    # Se marca RUNNING antes de lanzar la tarea: una segunda aprobación recibe 409 en vez de reanudar dos veces.
    await store.update(job_id, status=JobStatus.RUNNING, approval_request=None)
    _spawn(request.app, resume_job(request.app.state.graph, store, job_id, body.approved, body.comment))
    return {"job_id": job_id, "status": JobStatus.RUNNING}

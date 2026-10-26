"""Estado de los jobs en Redis.

Cada job se guarda como JSON en la clave `job:<id>` con un TTL, y el cliente de Redis es el
asíncrono (`redis.asyncio`) para no bloquear el event loop de FastAPI.
"""

import os
import time
import uuid
from enum import Enum
from typing import Any, Optional

import redis.asyncio as aioredis
from pydantic import BaseModel, Field

# Redis Stack (o Redis 8) en la base lógica 0, que es lo que exige el checkpointer de LangGraph.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))  # 24 horas


class JobStatus(str, Enum):
    PENDING = "PENDING"                    # encolado, todavía no arrancó
    RUNNING = "RUNNING"                    # el grafo se está ejecutando
    WAITING_APPROVAL = "WAITING_APPROVAL"  # pausado en el nodo Human-in-the-loop
    DONE = "DONE"                          # terminó y el reporte fue publicado
    REJECTED = "REJECTED"                  # un humano rechazó la acción crítica
    FAILED = "FAILED"                      # falló; el motivo queda en `error`


class Job(BaseModel):
    id: str
    query: str
    status: JobStatus = JobStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None  # "transient" (ej. 429) o "permanent"
    approval_request: Optional[dict[str, Any]] = None
    published: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class JobStore:
    def __init__(self, client: aioredis.Redis):
        self.client = client

    @classmethod
    def from_url(cls, url: str = REDIS_URL) -> "JobStore":
        return cls(aioredis.from_url(url, decode_responses=True))

    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    async def create(self, query: str) -> Job:
        job = Job(id=uuid.uuid4().hex, query=query)
        await self._save(job)
        return job

    async def get(self, job_id: str) -> Optional[Job]:
        raw = await self.client.get(self._key(job_id))
        return Job.model_validate_json(raw) if raw else None

    async def update(self, job_id: str, **fields: Any) -> Job:
        job = await self.get(job_id)
        if job is None:
            raise KeyError(f"El job {job_id} no existe (¿expiró?)")
        job = job.model_copy(update={**fields, "updated_at": time.time()})
        await self._save(job)
        return job

    async def _save(self, job: Job) -> None:
        await self.client.set(self._key(job.id), job.model_dump_json(), ex=JOB_TTL_SECONDS)

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def close(self) -> None:
        await self.client.aclose()

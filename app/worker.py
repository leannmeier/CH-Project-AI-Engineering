"""Tareas en segundo plano.

`run_job` ejecuta el grafo y `resume_job` lo reanuda tras la aprobación humana. Regla de oro
(error común que marca la consigna): cualquier excepción termina en FAILED con el motivo en
Redis. Si no, el cliente consultaría para siempre un job que quedó en RUNNING.
"""

import asyncio
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.hitl import publish_report
from app.jobs import JobStatus, JobStore

logger = logging.getLogger("app.worker")

MAX_ITERATIONS = 6
RECURSION_LIMIT = 20


def _thread_config(job_id: str) -> dict:
    # El thread_id es el id del job: así los checkpoints de Redis quedan asociados a cada job.
    return {"configurable": {"thread_id": job_id}, "recursion_limit": RECURSION_LIMIT}


def _initial_state(query: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "next_agent": "research_agent",
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS,
        "retry_count": 0,
        "research_data": [],
        "analysis_result": {},
        "task_status": "pending",
    }


def classify_error(message: str) -> str:
    """Errores pasajeros (límite de uso) frente a definitivos."""
    text = message.lower()
    return "transient" if "429" in text or "rate limit" in text else "permanent"


def _final_text(state: dict) -> str:
    messages = state.get("messages") or []
    return str(getattr(messages[-1], "content", "")) if messages else ""


def _failure_from_state(state: dict) -> tuple[str, str]:
    """Motivo (y tipo) del fallo, tomado del estado que dejaron los agentes."""
    research = state.get("research_data") or []
    if research and research[-1].get("status") == "error":
        last = research[-1]
        return last.get("error_message") or "La investigación falló.", last.get("error_type") or "permanent"
    analysis = state.get("analysis_result") or {}
    if analysis.get("status") == "error":
        message = analysis.get("error_message") or "El análisis falló."
        return message, classify_error(message)
    return "La tarea no pudo completarse.", "permanent"


async def _apply_result(store: JobStore, job_id: str, state: dict) -> None:
    """Traduce el resultado del grafo al estado del job en Redis."""
    interrupts = state.get("__interrupt__")
    if interrupts:
        # El grafo quedó pausado en el nodo de aprobación humana.
        first = interrupts[0]
        request = getattr(first, "value", first)
        await store.update(job_id, status=JobStatus.WAITING_APPROVAL, approval_request=request)
        return

    task_status = state.get("task_status")
    text = _final_text(state)

    if task_status == "failed":
        error, error_type = _failure_from_state(state)
        await store.update(job_id, status=JobStatus.FAILED, error=error, error_type=error_type)
    elif task_status == "rejected":
        await store.update(job_id, status=JobStatus.REJECTED, result=text, published=False)
    elif (state.get("approval") or {}).get("approved"):
        await publish_report(store, job_id, text)  # acción crítica, solo tras la aprobación
        await store.update(job_id, status=JobStatus.DONE, result=text, published=True)
    else:
        await store.update(
            job_id, status=JobStatus.FAILED, error="El grafo terminó sin pasar por la aprobación.",
            error_type="permanent",
        )


async def _fail(store: JobStore, job_id: str, message: str) -> None:
    try:
        await store.update(job_id, status=JobStatus.FAILED, error=message, error_type=classify_error(message))
    except Exception:  # si Redis también está caído, no hay nada más que hacer que registrarlo
        logger.exception("No se pudo registrar el fallo del job %s", job_id)


async def run_job(graph: Any, store: JobStore, job_id: str, query: str) -> None:
    try:
        await store.update(job_id, status=JobStatus.RUNNING)
        state = await graph.ainvoke(_initial_state(query), _thread_config(job_id))
        await _apply_result(store, job_id, state)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("El job %s falló", job_id)
        await _fail(store, job_id, f"{type(exc).__name__}: {exc}")


async def resume_job(
    graph: Any, store: JobStore, job_id: str, approved: bool, comment: Optional[str] = None
) -> None:
    try:
        decision = {"approved": approved, "comment": comment}
        state = await graph.ainvoke(Command(resume=decision), _thread_config(job_id))
        await _apply_result(store, job_id, state)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("El job %s falló al reanudar", job_id)
        await _fail(store, job_id, f"{type(exc).__name__}: {exc}")

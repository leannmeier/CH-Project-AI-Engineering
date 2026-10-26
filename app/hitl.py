"""Human-in-the-loop.

La acción crítica del sistema es PUBLICAR EL REPORTE. Antes de ejecutarla, el grafo se pausa en
`human_approval_node` con `interrupt()`. Queda esperando (con su estado guardado en el checkpointer)
hasta que alguien llama a `POST /tasks/{id}/approve`, que reanuda con `Command(resume=...)`.

OJO: al reanudar, LangGraph vuelve a ejecutar el nodo desde el principio. Por eso antes del
`interrupt()` no debe haber efectos secundarios: solo se calcula el borrador.
"""

import logging
from typing import Any

from langgraph.types import interrupt

from app.jobs import JOB_TTL_SECONDS, JobStore

logger = logging.getLogger("app.hitl")

CRITICAL_ACTION = "publicar_reporte"
MAX_DRAFT_CHARS = 3000


def _draft_from_state(state: dict) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    return str(getattr(messages[-1], "content", ""))[:MAX_DRAFT_CHARS]


async def human_approval_node(state: dict) -> dict:
    """Pausa el grafo hasta recibir la decisión humana sobre la acción crítica."""
    request: dict[str, Any] = {
        "action": CRITICAL_ACTION,
        "message": "Se requiere aprobación humana para publicar este reporte.",
        "draft": _draft_from_state(state),
    }

    # Pausa. Al reanudar, interrupt() devuelve lo que se envió en Command(resume=...)
    decision = interrupt(request)

    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        comment = decision.get("comment")
    else:
        approved, comment = bool(decision), None

    return {
        "approval": {"approved": approved, "comment": comment},
        "task_status": "completed" if approved else "rejected",
    }


async def publish_report(store: JobStore, job_id: str, report: str) -> None:
    """Acción crítica (simulada): 'publica' el reporte guardándolo en Redis."""
    await store.client.set(f"report:{job_id}", report, ex=JOB_TTL_SECONDS)
    logger.info("Reporte publicado para el job %s", job_id)

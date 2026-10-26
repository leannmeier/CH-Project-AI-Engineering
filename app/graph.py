"""Grafo de la API.

Reutiliza los nodos del Módulo 6 (supervisor, investigador, analista) y agrega un nodo de
aprobación humana antes de cerrar. El checkpointer es el de Redis (necesario para poder
pausar con `interrupt()` y retomar después).
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, Literal

from langgraph.graph import END, START, StateGraph

from app.hitl import human_approval_node
from app.jobs import REDIS_URL
from src.agents.analyst_agent import analyst_node
from src.agents.research_agent import research_node
from src.agents.supervisor import supervisor_node
from src.graph import route_decision as module6_route_decision
from state import MultiAgentState


class ApiState(MultiAgentState):
    """Estado del Módulo 6 más el resultado de la aprobación humana."""

    approval: Dict[str, Any]


def route(state: ApiState) -> Literal["research_agent", "analyst_agent", "human_approval", "__end__"]:
    """Igual que el ruteo del Módulo 6, pero un cierre exitoso pasa antes por la aprobación humana.

    Un cierre fallido (`failed`) termina directo: no tiene sentido pedir aprobación para publicar nada.
    """
    destination = module6_route_decision(state)
    if destination == "__end__" and state.get("task_status") != "failed":
        return "human_approval"
    return destination


def build_graph(checkpointer):
    workflow = StateGraph(ApiState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", research_node)
    workflow.add_node("analyst_agent", analyst_node)
    workflow.add_node("human_approval", human_approval_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route,
        {
            "research_agent": "research_agent",
            "analyst_agent": "analyst_agent",
            "human_approval": "human_approval",
            "__end__": END,
        },
    )
    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("analyst_agent", "supervisor")
    workflow.add_edge("human_approval", END)

    return workflow.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_checkpointer(url: str = REDIS_URL):
    """Abre el checkpointer de Redis una sola vez (al arrancar la API) y lo cierra al apagarla.

    Requiere Redis 8 o Redis Stack (módulos RedisJSON y RediSearch).
    """
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver  # import local: solo hace falta con Redis real

    async with AsyncRedisSaver.from_conn_string(url) as checkpointer:
        await checkpointer.asetup()  # crea los índices si todavía no existen (es idempotente)
        yield checkpointer

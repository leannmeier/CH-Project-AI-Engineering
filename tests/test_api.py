"""Tests de la API sin Redis real ni modelos: usan fakeredis, un checkpointer en memoria y nodos falsos.

Verifican la lógica propia de la entrega (estados de los jobs, pausa y reanudación con aprobación
humana, fallos que llegan a FAILED y concurrencia). No prueban Redis Stack, Phoenix ni Groq: eso se
verifica a mano (ver la lista de verificación del README).

Correr desde la raíz del repo:  pytest tests/test_api.py -v
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

os.environ["ENABLE_TRACING"] = "0"  # antes de importar app.main

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import app.graph as graph_module
from app.jobs import JobStore
from app.main import app


# ---------- nodos falsos (reemplazan a los agentes reales del Módulo 6) ----------

async def fake_research(state: dict) -> dict:
    query = state["messages"][0].content.lower()
    await asyncio.sleep(0.05)
    if "boom" in query:
        raise RuntimeError("Error 429 rate limit simulado")
    if "falla" in query:
        return {
            "messages": [],
            "research_data": [{
                "agent": "research_agent", "status": "error", "error_type": "permanent",
                "error_message": "La búsqueda web no devolvió resultados utilizables.", "data": "",
            }],
            "task_status": "failed",
        }
    return {
        "messages": [AIMessage(content=f"Informe sobre: {query}")],
        "research_data": [{"agent": "research_agent", "status": "success", "error_type": None,
                           "error_message": None, "data": "datos"}],
        "task_status": "pending",
    }


async def fake_analyst(state: dict) -> dict:
    return {
        "messages": [AIMessage(content="Promedio calculado: 42")],
        "analysis_result": {"agent": "analyst_agent", "status": "success", "tool_called": True,
                            "metrics": '{"average": 42}', "error_message": None},
        "task_status": "completed",
    }


async def fake_supervisor(state: dict) -> dict:
    research = state.get("research_data", [])
    analysis = state.get("analysis_result", {})
    query = state["messages"][0].content.lower()
    wants_calc = any(k in query for k in ["calcula", "promedio", "total", "suma"])
    iteration = state.get("iteration_count", 0) + 1
    if not research:
        return {"next_agent": "research_agent", "iteration_count": iteration}
    if research[-1].get("status") == "error":
        return {"next_agent": "__end__", "task_status": "failed"}
    if analysis:
        return {"next_agent": "__end__", "task_status": "completed"}
    if wants_calc:
        return {"next_agent": "analyst_agent", "iteration_count": iteration}
    return {"next_agent": "__end__", "task_status": "completed", "iteration_count": iteration}


# ---------- fixtures ----------

@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(graph_module, "supervisor_node", fake_supervisor)
    monkeypatch.setattr(graph_module, "research_node", fake_research)
    monkeypatch.setattr(graph_module, "analyst_node", fake_analyst)

    monkeypatch.setattr(
        JobStore, "from_url", classmethod(lambda cls, url=None: cls(FakeAsyncRedis(decode_responses=True)))
    )

    @asynccontextmanager
    async def memory_checkpointer(url=None):
        yield InMemorySaver()

    monkeypatch.setattr("app.main.open_checkpointer", memory_checkpointer)

    with TestClient(app) as test_client:
        yield test_client


def wait_for(client: TestClient, job_id: str, statuses: set[str], timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        job = client.get(f"/tasks/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"El job {job_id} no llegó a {statuses}; último estado: {job.get('status')}")


# ---------- tests ----------

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_post_devuelve_job_id_al_instante(client):
    response = client.post("/tasks", json={"query": "Informe sobre satélites"})
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] and body["status"] in {"PENDING", "RUNNING"}


def test_flujo_completo_con_aprobacion(client):
    job_id = client.post("/tasks", json={"query": "Informe sobre satélites"}).json()["job_id"]

    waiting = wait_for(client, job_id, {"WAITING_APPROVAL"})
    assert waiting["approval_request"]["action"] == "publicar_reporte"
    assert "Informe sobre" in waiting["approval_request"]["draft"]
    assert waiting["published"] is False  # nada se publica antes de la aprobación

    assert client.post(f"/tasks/{job_id}/approve", json={"approved": True, "comment": "ok"}).status_code == 202

    done = wait_for(client, job_id, {"DONE"})
    assert done["published"] is True
    assert "Informe sobre" in done["result"]


def test_flujo_con_calculo_pasa_por_el_analista(client):
    job_id = client.post("/tasks", json={"query": "Busca datos y calcula el promedio"}).json()["job_id"]
    waiting = wait_for(client, job_id, {"WAITING_APPROVAL"})
    assert "Promedio calculado" in waiting["approval_request"]["draft"]


def test_rechazo_no_publica(client):
    job_id = client.post("/tasks", json={"query": "Informe sobre satélites"}).json()["job_id"]
    wait_for(client, job_id, {"WAITING_APPROVAL"})
    client.post(f"/tasks/{job_id}/approve", json={"approved": False, "comment": "no"})
    rejected = wait_for(client, job_id, {"REJECTED"})
    assert rejected["published"] is False


def test_fallo_del_agente_llega_a_failed_con_motivo(client):
    job_id = client.post("/tasks", json={"query": "esto falla"}).json()["job_id"]
    failed = wait_for(client, job_id, {"FAILED"})
    assert "no devolvió resultados" in failed["error"]
    assert failed["error_type"] == "permanent"


def test_excepcion_no_deja_el_job_colgado(client):
    job_id = client.post("/tasks", json={"query": "esto hace boom"}).json()["job_id"]
    failed = wait_for(client, job_id, {"FAILED"})
    assert "429" in failed["error"]
    assert failed["error_type"] == "transient"


def test_aprobar_job_inexistente_o_en_mal_estado(client):
    assert client.post("/tasks/no-existe/approve", json={"approved": True}).status_code == 404
    job_id = client.post("/tasks", json={"query": "Informe sobre satélites"}).json()["job_id"]
    wait_for(client, job_id, {"WAITING_APPROVAL"})
    client.post(f"/tasks/{job_id}/approve", json={"approved": True})
    wait_for(client, job_id, {"DONE"})
    assert client.post(f"/tasks/{job_id}/approve", json={"approved": True}).status_code == 409


def test_consulta_invalida_devuelve_422(client):
    assert client.post("/tasks", json={"query": "a"}).status_code == 422


def test_cinco_jobs_concurrentes(client):
    job_ids = [client.post("/tasks", json={"query": f"Informe sobre tema {i}"}).json()["job_id"] for i in range(5)]
    for job_id in job_ids:
        wait_for(client, job_id, {"WAITING_APPROVAL"})
    assert len(set(job_ids)) == 5

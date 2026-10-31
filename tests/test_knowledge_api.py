"""Pruebas de contrato del endpoint RAG sin conexión a Pinecone."""

import pytest

from tests.test_api import client  # reutiliza la API con Redis y nodos simulados
from app.knowledge import KnowledgeHit, KnowledgeResponse


def test_knowledge_search_returns_sources(client, monkeypatch):
    import app.main as api_module

    async def fake_search(question, k):
        return KnowledgeResponse(
            question=question,
            hits=[KnowledgeHit(source="data/pathlib/01-rutas.txt", content="Path permite crear rutas")],
        )

    monkeypatch.setattr(api_module, "search_knowledge", fake_search)
    response = client.post("/knowledge/search", json={"question": "Cómo construir una ruta?"})
    assert response.status_code == 200
    assert response.json()["hits"][0]["source"] == "data/pathlib/01-rutas.txt"


def test_knowledge_search_rejects_invalid_request(client):
    assert client.post("/knowledge/search", json={"question": "a"}).status_code == 422
    assert client.post("/knowledge/search", json={"question": "consulta válida", "k": 6}).status_code == 422


def test_knowledge_search_reports_unavailable_service(client, monkeypatch):
    import app.main as api_module

    async def fake_failure(question, k):
        raise RuntimeError("error de backend simulado")

    monkeypatch.setattr(api_module, "search_knowledge", fake_failure)
    response = client.post("/knowledge/search", json={"question": "consulta técnica"})
    assert response.status_code == 503

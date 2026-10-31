"""Regresiones del Módulo 4; no requieren Pinecone, Redis ni claves API."""

import pytest
from langchain_core.documents import Document

from evaluate import GOLDEN_SET, normalize_source, score_retrieval, run_evaluation
from rag_chain import RAGSystem


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs

    async def ainvoke(self, query):
        return self.docs


def test_rag_system_enforces_top_five_after_hybrid_fusion():
    docs = [Document(page_content=str(i), metadata={"source": f"doc{i}"}) for i in range(9)]
    system = RAGSystem(k=5, retriever=FakeRetriever(docs))
    assert len(system.retrieve("consulta")) == 5
    assert [doc.page_content for doc in system.retrieve("consulta")] == [str(i) for i in range(5)]


@pytest.mark.asyncio
async def test_rag_system_async_top_five():
    docs = [Document(page_content=str(i)) for i in range(8)]
    system = RAGSystem(k=5, retriever=FakeRetriever(docs))
    assert len(await system.aretrieve("consulta")) == 5
    assert len(await system.as_runnable().ainvoke("consulta")) == 5


def test_recall_and_precision_use_source_not_keywords():
    docs = [
        Document(page_content="No se menciona ningún término del enunciado", metadata={"source": "data/pathlib/01-rutas.txt"}),
        Document(page_content="Path, Path y más Path", metadata={"source": "data/pathlib/01-rutas.txt"}),
        Document(page_content="Path", metadata={"source": "data/pathlib/02-comprobaciones.txt"}),
    ]
    recall, precision = score_retrieval(docs, ["data/pathlib/01-rutas.txt"])
    assert recall == 1.0
    assert precision == 2 / 5


def test_source_normalization_handles_absolute_and_windows_style():
    assert normalize_source("/repo/data/pathlib/01-rutas.txt") == "data/pathlib/01-rutas.txt"
    assert normalize_source(r"C:\\repo\\data\\pathlib\\01-rutas.txt") == "data/pathlib/01-rutas.txt"


def test_missing_source_returns_no_hits():
    docs = [Document(page_content="Texto", metadata={})]
    assert score_retrieval(docs, ["data/pathlib/01-rutas.txt"]) == (0.0, 0.0)


def test_golden_set_has_five_distinct_existing_sources():
    from pathlib import Path
    assert len(GOLDEN_SET) == 5
    expected_sources = [source for case in GOLDEN_SET for source in case["expected_sources"]]
    assert len(expected_sources) == len(set(expected_sources)) == 5
    assert all(Path(source).is_file() for source in expected_sources)


def test_evaluation_accepts_injected_retriever_without_network():
    class FakeRAG:
        def retrieve(self, query):
            case = next(c for c in GOLDEN_SET if c["query"] == query)
            return [Document(page_content="Información útil", metadata={"source": case["expected_sources"][0]})]

    results = run_evaluation(rag_system=FakeRAG())
    assert results["questions"] == 5
    assert results["mean_recall@5"] == 1.0
    assert results["mean_precision@5"] == 0.2

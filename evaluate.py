"""Benchmark RAG reproducible: cinco consultas con fuentes conocidas.

Ejecutar después de indexar `data/pathlib/` en Pinecone:
    python -c "from vectorstore import build_and_index_vectorstore; build_and_index_vectorstore()"
    python evaluate.py

Recall@5: fracción de fuentes relevantes conocidas presentes entre los cinco resultados.
Precision@5: fracción de los cinco resultados cuya metadata source corresponde
a una fuente relevante para esa pregunta. Se evalúan CHUNKS (puede haber varios
del mismo archivo), y el denominador se mantiene en 5.
"""

import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from rag_chain import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Cada pregunta tiene su documento fuente verificable en el dataset del repositorio.
GOLDEN_SET = [
    {
        "query": "¿Cómo puedo componer una ruta con el operador / de pathlib.Path?",
        "expected_sources": ["data/pathlib/01-rutas.txt"],
    },
    {
        "query": "¿Qué métodos de pathlib verifican si una ruta existe y si es un archivo?",
        "expected_sources": ["data/pathlib/02-comprobaciones.txt"],
    },
    {
        "query": "¿Cómo leo y escribo un archivo de texto UTF-8 usando pathlib?",
        "expected_sources": ["data/pathlib/03-lectura-escritura.txt"],
    },
    {
        "query": "¿Qué diferencia hay entre glob, rglob e iterdir en pathlib?",
        "expected_sources": ["data/pathlib/04-busqueda.txt"],
    },
    {
        "query": "¿Cómo se crean directorios padre y cómo se elimina un archivo con Path?",
        "expected_sources": ["data/pathlib/05-operaciones.txt"],
    },
]


def normalize_source(source: str) -> str:
    """Normaliza separadores, rutas relativas y absolutas dentro de data/."""
    path = Path(source).as_posix().replace("\\", "/")
    marker = "data/"
    if marker in path:
        return marker + path.rsplit(marker, 1)[-1]
    return path.lstrip("./")


def score_retrieval(
    retrieved_docs: Iterable[Document],
    expected_sources: Iterable[str],
    k: int = 5,
) -> tuple[float, float]:
    """Calcula recall de documentos fuente y precision de los k chunks recuperados.

    El recall usa fuentes únicas (no cuenta múltiples chunks del mismo archivo
    como nuevas fuentes relevantes). La precisión cuenta cada chunk relevante
    individualmente dentro de los primeros k resultados.
    """
    if k < 1:
        raise ValueError("k debe ser mayor que cero")
    expected = {normalize_source(source) for source in expected_sources}
    if not expected:
        raise ValueError("Se necesita al menos una fuente esperada")

    docs = list(retrieved_docs)[:k]
    retrieved = [normalize_source(str(doc.metadata.get("source", ""))) for doc in docs]
    recall = len(expected.intersection(retrieved)) / len(expected)
    precision = sum(source in expected for source in retrieved) / k
    return recall, precision


def run_evaluation(k: int = 5, rag_system: RAGSystem | None = None) -> dict:
    """Ejecuta el benchmark y devuelve métricas agregadas además de imprimirlas."""
    rag_system = rag_system or RAGSystem(k=k)
    recalls, precisions = [], []

    print(f"\n--- EVALUACIÓN CUANTITATIVA (Top-{k}) ---\n")
    for test in GOLDEN_SET:
        docs = rag_system.retrieve(test["query"])[:k]
        recall, precision = score_retrieval(docs, test["expected_sources"], k=k)
        recalls.append(recall)
        precisions.append(precision)
        actual_sources = [doc.metadata.get("source", "(sin source)") for doc in docs]
        print(f"Pregunta: {test['query']}")
        print(f"Fuente esperada: {', '.join(test['expected_sources'])}")
        print(f"Fuentes recuperadas: {actual_sources}")
        print(f"Recall@{k}: {recall:.2f} | Precision@{k}: {precision:.2f}\n")

    results = {
        "questions": len(GOLDEN_SET),
        f"mean_recall@{k}": sum(recalls) / len(recalls),
        f"mean_precision@{k}": sum(precisions) / len(precisions),
    }
    print("RESULTADOS GLOBALES:")
    print(f"Mean Recall@{k}: {results[f'mean_recall@{k}']:.2%}")
    print(f"Mean Precision@{k}: {results[f'mean_precision@{k}']:.2%}")
    return results


if __name__ == "__main__":
    run_evaluation(k=5)

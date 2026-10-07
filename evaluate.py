import asyncio
import logging
from rag_chain import build_hybrid_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation(k: int = 5):
    # Benchmark / Golden Set de prueba
    benchmark = [
        {
            "query": "¿Qué librería se menciona para validación de datos?",
            "expected_keyword": "Pydantic",
        },
        {
            "query": "¿Qué base de datos se utiliza en la nube?",
            "expected_keyword": "Pinecone",
        },
    ]

    retriever = build_hybrid_retriever(k=k)
    recalls, precisions = [], []

    print(f"\n--- INICIANDO EVALUACIÓN CUANTITATIVA (Top-{k}) ---\n")

    for test in benchmark:
        query = test["query"]
        expected = test["expected_keyword"].lower()

        retrieved_docs = retriever.invoke(query)

        # Hit de relevancia si la palabra clave esperada está en los fragmentos recuperados
        hits = [
            doc
            for doc in retrieved_docs
            if expected in doc.page_content.lower()
        ]
        is_relevant = len(hits) > 0

        recall = 1.0 if is_relevant else 0.0
        precision = len(hits) / len(retrieved_docs) if retrieved_docs else 0.0

        recalls.append(recall)
        precisions.append(precision)

        print(f"Pregunta: {query}")
        print(f" Relevancia hallada: {is_relevant}")
        print(f" Recall@{k}: {recall:.2f} | Precision@{k}: {precision:.2f}\n")

    mean_recall = sum(recalls) / len(recalls)
    mean_precision = sum(precisions) / len(precisions)

    print("RESULTADOS GLOBALES:")
    print(f"Mean Recall@{k}: {mean_recall:.2%}")
    print(f"Mean Precision@{k}: {mean_precision:.2%}")


if __name__ == "__main__":
    run_evaluation(k=5)
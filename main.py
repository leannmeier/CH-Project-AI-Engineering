import asyncio
import sys

from evaluate import run_evaluation
from rag_chain import query_rag
from vectorstore import build_and_index_vectorstore


async def main():
    print("1. Verificando/Indexando documentos hacia Pinecone Cloud")
    await asyncio.to_thread(build_and_index_vectorstore)

    print(
        "\n2. Probando la Cadena RAG Híbrida (Pinecone + BM25) con Pydantic"
    )
    pregunta = "¿Qué información contiene el documento cargado?"

    try:
        resultado = await query_rag(pregunta)
        print(f"Pregunta: {resultado.question}")
        print(f"Respuesta: {resultado.answer}\n")
        print("Fuentes Utilizadas (Trazabilidad Real):")
        for src in resultado.sources:
            print(f"- Archivo: {src.source_file}")
            print(f"  Snippet: {src.content_snippet}")
            print(f"  Metadatos: {src.metadata}\n")
    except Exception as e:
        print(f"\nError durante la ejecución del RAG: {e}")


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        run_evaluation(k=5)
    else:
        asyncio.run(main())
import asyncio
from vectorstore import build_and_index_vectorstore
from rag_chain import query_rag


async def main():
    print("--- 1. Verificando/Indexando documentos en ./data ---")
    # Ejecuta I/O de indexación
    await asyncio.to_thread(build_and_index_vectorstore)

    print("\n--- 2. Probando la Cadena RAG con pregunta sobre el documento ---")
    pregunta = "¿Qué información contiene el documento cargado?"
    
    try:
        resultado = await query_rag("¿Qué información contiene el documento cargado?")
        print(f"Pregunta: {resultado.question}")
        print(f"Respuesta: {resultado.answer}\n")
        print("--- Fuentes Utilizadas (Trazabilidad Real) ---")
        for src in resultado.sources:
            print(f"- Archivo: {src.source_file}")
            print(f"  Snippet: {src.content_snippet}")
            print(f"  Metadatos: {src.metadata}\n")
    except Exception as e:
        print(f"\nError durante la ejecución del RAG: {e}")

if __name__ == "__main__":
    asyncio.run(main())
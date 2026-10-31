"""Acceso asíncrono al RAG híbrido desde la API unificada.

No genera respuestas con un LLM: recupera fragmentos fundamentados y sus
fuentes. El LLM y el sistema multi-agente continúan siendo componentes separados.
El recuperador se construye en un hilo para no bloquear el event loop mientras
carga embeddings locales y conecta con Pinecone.
"""

import asyncio
from typing import List

from pydantic import BaseModel, Field


class KnowledgeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=5)


class KnowledgeHit(BaseModel):
    source: str
    content: str
    metadata: dict = Field(default_factory=dict)


class KnowledgeResponse(BaseModel):
    question: str
    hits: List[KnowledgeHit] = Field(default_factory=list)


async def search_knowledge(question: str, k: int = 5) -> KnowledgeResponse:
    # El import diferido evita cargar Pinecone/HuggingFace para iniciar la API
    # y permite utilizar los endpoints de los módulos 6-7 de forma independiente.
    from rag_chain import RAGSystem

    rag = await asyncio.to_thread(RAGSystem, k=k)
    docs = await rag.aretrieve(question)
    hits = [
        KnowledgeHit(
            source=str(doc.metadata.get("source", "Desconocido")),
            content=doc.page_content,
            metadata=doc.metadata,
        )
        for doc in docs[:k]
    ]
    return KnowledgeResponse(question=question, hits=hits)

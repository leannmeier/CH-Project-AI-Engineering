import asyncio
import logging
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

from Config import Config, ClientType
from factory import LLMFactory
from vectorstore import get_vectorstore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- 1. Modelos Pydantic para Entrada y Salida ---

class RAGQueryInput(BaseModel):
    """Modelo de entrada con validación de longitud mínima de la consulta."""
    question: str = Field(..., min_length=3, description="La consulta del usuario.")


class DocumentSource(BaseModel):
    """Modelo de trazabilidad para cada fragmento recuperado."""
    source_file: str = Field(..., description="Ruta o nombre del archivo origen.")
    content_snippet: str = Field(..., description="Fragmento del contenido utilizado.")
    metadata: dict = Field(default_factory=dict, description="Metadatos completos del chunk.")


class RAGQueryOutput(BaseModel):
    """Modelo de salida que incluye la respuesta del LLM y las fuentes reales con trazabilidad."""
    question: str
    answer: str
    sources: List[DocumentSource] = Field(
        default_factory=list, 
        description="Lista de fuentes recuperadas por el retriever para esta consulta."
    )


# --- 2. Prompts y Auxiliares ---

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un asistente experto para responder preguntas basadas en contexto.\n"
        "Responde la pregunta del usuario utilizando únicamente el siguiente contexto recuperado.\n"
        "Si no sabes la respuesta o no está en el contexto, di claramente 'No tengo suficiente información en los documentos para responder esa pregunta'.\n\n"
        "Contexto recuperado:\n{context}"
    ),
    ("user", "{question}")
])


def format_docs(docs: List[Document]) -> str:
    """Concatena el texto de los documentos para el prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: List[Document]) -> List[DocumentSource]:
    """Transforma los objetos Document de LangChain en modelos de Pydantic con trazabilidad real."""
    sources = []
    for doc in docs:
        source_path = doc.metadata.get("source", "Desconocido")
        sources.append(
            DocumentSource(
                source_file=source_path,
                content_snippet=doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                metadata=doc.metadata
            )
        )
    return sources


# --- 3. Construcción del Pipeline LCEL Paralelelizado ---

def build_rag_chain(llm: BaseChatModel, retriever):
    """
    Construye la cadena RAG preservando tanto los Documentos originales (fuentes)
    como la cadena de generación usando RunnableParallel.
    """
    # Ramificación paralela 1: El retriever recupera los objetos Document
    retrieval_chain = RunnableParallel({
        "docs": retriever,
        "question": RunnablePassthrough()
    })

    # Ramificación paralela 2: Formateo de contexto para el prompt + Generación del LLM
    generation_chain = (
        {
            "context": lambda x: format_docs(x["docs"]),
            "question": lambda x: x["question"]
        }
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    # Cadena combinada LCEL que retorna tanto la respuesta en texto como los docs recuperados
    full_chain = retrieval_chain | RunnableParallel({
        "answer": generation_chain,
        "docs": lambda x: x["docs"],
        "question": lambda x: x["question"]
    })

    return full_chain


# --- 4. Función de Ejecución Asíncrona Principal ---

async def query_rag(raw_question: str, config: Optional[Config] = None, llm: Optional[BaseChatModel] = None) -> RAGQueryOutput:
    """
    Ejecuta la consulta RAG validando E/S con Pydantic y sin bloqueos de I/O.
    """
    # Validar entrada
    query_input = RAGQueryInput(question=raw_question)
    logger.info(f"Procesando consulta RAG validada: '{query_input.question}'...")

    if config is None and llm is None:
        config = LLMFactory.get_default_config(ClientType.OPENAI)

    if llm is None:
        llm = LLMFactory.create_llm(config)

    # Operación de I/O no bloqueante para instanciar el retriever
    def get_retriever_sync():
        vectorstore = get_vectorstore(config)
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever = await asyncio.to_thread(get_retriever_sync)

    # Construir cadena LCEL
    chain = build_rag_chain(llm, retriever)

    try:
        # Invocar la cadena en paralelo
        raw_result = await chain.ainvoke(query_input.question)
        
        # Extraer y estructurar fuentes reales
        validated_sources = extract_sources(raw_result["docs"])

        # Construir y retornar el objeto Pydantic validado con trazabilidad
        response_output = RAGQueryOutput(
            question=query_input.question,
            answer=raw_result["answer"],
            sources=validated_sources
        )
        
        logger.info("Consulta RAG completada con trazabilidad y fuentes reales.")
        return response_output

    except Exception as e:
        logger.error(f"Error en la ejecución de la cadena RAG: {e}")
        raise
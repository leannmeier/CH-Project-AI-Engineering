import os
import logging
from typing import Optional
from dotenv import load_dotenv

from langchain_chroma import Chroma
from factory import LLMFactory
from Config import Config
from ingest import load_documents, split_documents

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_db"


def get_vectorstore(config: Optional[Config] = None) -> Chroma:
    """Obtiene o inicializa la instancia de ChromaDB persistente."""
    embedding_function = LLMFactory.create_embeddings(config)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )


def build_and_index_vectorstore(config: Optional[Config] = None) -> Chroma:
    """Indexa documentos en ChromaDB solo si el vectorstore está vacío."""
    vectorstore = get_vectorstore(config)
    
    # Control de duplicación: verificar si ya hay vectores almacenados
    existing_docs = vectorstore.get()
    if existing_docs and existing_docs.get("ids"):
        logger.info(f"El VectorStore en '{CHROMA_PATH}' ya contiene datos. Omitiendo reindexación para evitar duplicados.")
        return vectorstore

    logger.info("Iniciando proceso de lectura e indexación de documentos...")
    raw_documents = load_documents()
    if not raw_documents:
        logger.warning("No se encontraron documentos en la carpeta ./data.")
        return vectorstore

    chunks = split_documents(raw_documents)
    embedding_function = LLMFactory.create_embeddings(config)

    logger.info(f"Guardando {len(chunks)} chunks en ChromaDB ('{CHROMA_PATH}')...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    logger.info("¡Indexación completada exitosamente!")
    return vectorstore
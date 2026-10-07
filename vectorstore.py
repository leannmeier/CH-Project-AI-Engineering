import logging
import os
import time
from typing import Optional
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from Config import Config
from factory import LLMFactory
from ingest import load_documents, split_documents

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INDEX_NAME = os.getenv("INDEX_NAME", "rag-hybrid-index")


def get_vectorstore(config: Optional[Config] = None) -> PineconeVectorStore:
    """Obtiene la instancia del VectorStore remoto en Pinecone"""
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    existing_indexes = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        logger.info(
            f"Creando indice Serverless '{INDEX_NAME}' en Pinecone..."
        )
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    embedding_function = LLMFactory.create_embeddings(config)
    return PineconeVectorStore(
        index_name=INDEX_NAME, embedding=embedding_function
    )


def build_and_index_vectorstore(
    config: Optional[Config] = None,
) -> PineconeVectorStore:
    """Lee documentos locales y los sube a Pinecone Serverless"""
    vectorstore = get_vectorstore(config)

    logger.info(
        "Iniciando proceso de lectura e indexación de documentos para Pinecone..."
    )
    raw_documents = load_documents()
    if not raw_documents:
        logger.warning("No se encontraron documentos en la carpeta ./data")
        return vectorstore

    chunks = split_documents(raw_documents)
    embedding_function = LLMFactory.create_embeddings(config)

    logger.info(
        f"Subiendo {len(chunks)} chunks a Pinecone Cloud ('{INDEX_NAME}')..."
    )
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embedding_function, index_name=INDEX_NAME
    )
    logger.info("Indexación en la nube completada exitosamente!!!")
    return vectorstore
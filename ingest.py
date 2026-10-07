import os
import logging
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "./data"

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """
    Carga todos los documentos soportados (PDF y TXT) desde el directorio especificado.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        logger.warning(f"La carpeta '{data_path}' no existía. Se ha creado. Coloca tus archivos allí.")
        return []

    logger.info(f"Cargando documentos desde '{data_path}'...")
    documents = []

    # Cargador para archivos .pdf
    pdf_loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents.extend(pdf_loader.load())

    # Cargador para archivos .txt
    txt_loader = DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    
    documents.extend(txt_loader.load())

    logger.info(f"Se cargaron {len(documents)} páginas/documentos en total.")
    return documents


def split_documents(
    documents: List[Document], 
    chunk_size: int = 800, 
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Fragmenta los documentos cargados en chunks pequeños optimizados para RAG.
    """
    logger.info(f"Fragmentando documentos (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documentos divididos en {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        print(f"\nEjemplo del primer chunk de {len(chunks)}")
        print("Contenido:", chunks[0].page_content[:200])
        print(f"\nMetadatos:", chunks[0].metadata)
    else:
        print("Agrega archivos .pdf o .txt en la carpeta './data' para probar la ingesta.")
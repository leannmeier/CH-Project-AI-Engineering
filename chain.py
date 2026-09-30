import logging
from typing import Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from Config import Config, ClientType
from factory import LLMFactory
from schemas import AnalisisTecnico

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Prompt base desacoplado
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un analista tecnico de sistemas experto. Tu tarea es analizar el texto "
        "ingresado, identificar las tecnologías involucradas, determinar el nivel de criticidad "
        "(baja, media o alta) y generar un resumen tecnico preciso"
    ),
    ("user", "{input_text}")
])

def build_chain(llm: BaseChatModel):
    """Construye un pipeline LCEL desacoplado, validado y resiliente a partir de un LLM."""
    structured_llm = llm.with_structured_output(AnalisisTecnico)
    
    # Resiliencia: 3 reintentos automáticos
    resilient_llm = structured_llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )
    
    return PROMPT_TEMPLATE | resilient_llm

async def process_text(text: str, config: Optional[Config] = None, llm: Optional[BaseChatModel] = None) -> AnalisisTecnico:
    """
    De manera asincrona, procesa un texto.
    Permite inyectar un `llm` o un `config` personalizado. Si no se especifican, 
    utiliza la configuración por defecto de la LLMFactory.
    """
    logger.info("Iniciando procesamiento asíncrono...")
    # Aca hacemos la inyección de dependencias
    if llm is None:
        if config is None:
            config = LLMFactory.get_default_config(ClientType.OPENAI)
        llm = LLMFactory.create_llm(config)

    # Construcción dinámica de la cadena para la invocación
    chain = build_chain(llm)
    try:
        result: AnalisisTecnico = await chain.ainvoke({"input_text": text})
        logger.info("Procesamiento completado y validado con éxito.")
        return result
    except Exception as e:
        logger.error(f"Error en la ejecución del pipeline: {e}")
        raise
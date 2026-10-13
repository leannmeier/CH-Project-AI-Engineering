import json
 
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent
 
from Config import ClientType
from factory import LLMFactory
from src.tools import RESEARCH_TOOLS
 
MAX_CHARS_PER_RESULT = 600
MAX_CHARS_TOTAL = 4000
 
SYSTEM_PROMPT = (
    "Eres un Agente de Investigación web experto. Utiliza Tavily para buscar información oficial.\n"
    "REQUISITO DE AUDITORÍA: para cada cifra numérica que recopiles, identifica claramente "
    "la entidad, el valor, el año de referencia y la fuente/URL de donde se obtuvo.\n"
    "Realiza como máximo 3 búsquedas y cierra con lo que hayas reunido."
)
 
 
def _is_tool_message(msg) -> bool:
    return isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool"
 
 
def _format_tavily_content(content) -> list[str]:
    """Convierte el contenido de un ToolMessage de Tavily en bloques de texto limpios (sin 'score')."""
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return [content] if content.strip() else []
    if isinstance(content, dict):
        content = content.get("results", [])
 
    blocks = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("content", ""))[:MAX_CHARS_PER_RESULT]
                blocks.append(
                    f"- Título: {item.get('title', 'N/A')}\n"
                    f"  URL: {item.get('url', 'N/A')}\n"
                    f"  Contenido: {text}"
                )
    return blocks
 
 
def _error_result(messages: list, error_type: str, error_message: str) -> dict:
    return {
        "messages": messages,
        "research_data": [{
            "agent": "research_agent",
            "tool_used": "tavily_search",
            "status": "error",
            "error_type": error_type,
            "error_message": error_message,
            "data": "",
        }],
        "task_status": "failed",
    }
 
 
async def research_node(state: dict) -> dict:
    llm = LLMFactory.create_llm(LLMFactory.get_default_config(provider=ClientType.GROQ))
    agent = create_react_agent(llm, tools=RESEARCH_TOOLS, prompt=SYSTEM_PROMPT)
    input_messages = state.get("messages", [])
 
    try:
        # Límite de pasos propio: si el agente se cicla, falla rápido y gasta menos tokens
        result = await agent.ainvoke(
            {"messages": input_messages}, config={"recursion_limit": 10}
        )
        res_messages = result["messages"]
        new_messages = res_messages[len(input_messages):]
 
        tool_messages = [
            m for m in res_messages
            if _is_tool_message(m) and getattr(m, "status", "success") != "error"
        ]
        blocks = []
        for tm in tool_messages:
            blocks.extend(_format_tavily_content(tm.content))
 
        print(f"🔍 [Research Agent]: {len(tool_messages)} búsqueda(s) con resultado, {len(blocks)} fuente(s) útiles.")
 
        if not blocks:
            return _error_result(
                new_messages, "permanent", "La búsqueda web no devolvió resultados utilizables."
            )
 
        return {
            "messages": new_messages,
            "research_data": [{
                "agent": "research_agent",
                "tool_used": "tavily_search",
                "status": "success",
                "error_type": None,
                "error_message": None,
                "data": "\n".join(blocks)[:MAX_CHARS_TOTAL],
            }],
            "task_status": "pending",
        }
 
    except Exception as e:
        err = str(e)
        error_type = "transient" if "429" in err or "rate limit" in err.lower() else "permanent"
        print(f"❌ [Research Error]: {err}")
        return _error_result([], error_type, f"Excepción en investigación: {err}")
 
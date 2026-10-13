import json
 
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
 
from Config import ClientType
from factory import LLMFactory
from src.tools import ANALYST_TOOLS
 
 
def _valid_metrics(msg) -> bool:
    """True si el mensaje es un resultado válido de calculate_metrics (sin error)."""
    if not (isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool"):
        return False
    if getattr(msg, "status", "success") == "error":
        return False
    try:
        data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
    except ValueError:
        return False
    return isinstance(data, dict) and "error" not in data and "average" in data
 
 
def _analysis(status: str, metrics, error_message=None) -> dict:
    return {
        "agent": "analyst_agent",
        "status": status,
        "tool_called": metrics is not None,
        "metrics": metrics,
        "error_message": error_message,
    }
 
 
async def analyst_node(state: dict) -> dict:
    research_data = state.get("research_data", [])
    raw_research = research_data[-1].get("data", "") if research_data else ""
    messages_history = state.get("messages", [])
    user_query = messages_history[0].content if messages_history else ""
 
    llm = LLMFactory.create_llm(LLMFactory.get_default_config(provider=ClientType.GROQ))
 
    system_prompt = (
        "Eres el Agente de Análisis y Cómputo experto.\n"
        f"Pregunta original: '{user_query}'\n"
        "INSTRUCCIONES DE AUDITORÍA Y CÁLCULO:\n"
        "1. Selecciona rigurosamente el mismo nivel administrativo para todas las entidades "
        "(ej. todas ciudades). Si falta el dato de alguna entidad en ese nivel, dilo explícitamente "
        "en lugar de reemplazarlo por otro nivel.\n"
        "2. Invoca obligatoriamente la herramienta `calculate_metrics` con los valores numéricos limpios.\n"
        "3. En tu informe final, cita explícitamente el año y la fuente/URL de cada cifra utilizada."
    )
 
    agent = create_react_agent(llm, tools=ANALYST_TOOLS, prompt=system_prompt)
    agent_input = [HumanMessage(content=f"Datos de investigación con fuentes:\n{raw_research}")]
 
    try:
        result = await agent.ainvoke(
            {"messages": agent_input}, config={"recursion_limit": 8}
        )
        messages = result["messages"]
        new_messages = messages[len(agent_input):]  # excluye el mensaje de entrada
 
        valid = [m for m in messages if _valid_metrics(m)]
        print(f"🧮 [Analyst Agent]: métricas válidas obtenidas: {bool(valid)}")
 
        if not valid:
            return {
                "messages": new_messages,
                "analysis_result": _analysis(
                    "error", None, "El analista no obtuvo métricas válidas de la herramienta de cálculo."
                ),
                "task_status": "failed",
            }
 
        return {
            "messages": new_messages,
            "analysis_result": _analysis("success", valid[-1].content),
            "task_status": "completed",
        }
 
    except Exception as e:
        print(f"❌ [Analyst Error]: {e}")
        return {
            "messages": [],
            "analysis_result": _analysis("error", None, str(e)),
            "task_status": "failed",
        }
 
from typing import Literal
 
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
 
from Config import ClientType
from factory import LLMFactory
 
CALC_KEYWORDS = ["calculá", "calcula", "promedio", "total", "suma", "métrica"]
 
SYSTEM_PROMPT = (
    "Eres el supervisor de un sistema multi-agente. Ya existe una investigación exitosa "
    "y todavía no hay análisis. Elige el siguiente paso:\n"
    "- 'analyst_agent': si el usuario pidió cálculos o métricas.\n"
    "- '__end__': si la consulta ya quedó respondida con la investigación.\n"
    "Explica tu decisión en español."
)
 
 
class SupervisorOutput(BaseModel):
    next_agent: Literal["research_agent", "analyst_agent", "__end__"] = Field(
        description="Siguiente paso: 'research_agent', 'analyst_agent' o '__end__'."
    )
    reason: str = Field(description="Explicación breve, en español, de la decisión de ruteo.")
 
 
async def supervisor_node(state: dict) -> dict:
    research_data = state.get("research_data", [])
    analysis_result = state.get("analysis_result", {})
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 6)
 
    messages = state.get("messages", [])
    user_query = messages[0].content if messages else ""
    requires_calculation = any(k in user_query.lower() for k in CALC_KEYWORDS)
 
    # 1. Tope de iteraciones
    if iteration_count >= max_iterations:
        print("⚠️ [Supervisor]: Límite de iteraciones alcanzado. Cierre forzado.")
        return {"next_agent": "__end__", "task_status": "failed"}
 
    # 2. Error en la investigación: se reintenta una vez solo si es pasajero (ej. 429)
    if research_data and research_data[-1].get("status") == "error":
        last = research_data[-1]
        retries = state.get("retry_count", 0)
        if last.get("error_type") == "transient" and retries < 1:
            print("🔄 [Supervisor]: Error pasajero. Reintentando investigación (1/1)...")
            return {
                "next_agent": "research_agent",
                "retry_count": retries + 1,
                "iteration_count": iteration_count + 1,
                "task_status": "pending",
            }
        print(f"🛡️ [Supervisor]: La investigación falló ({last.get('error_message')}). Cierre.")
        return {"next_agent": "__end__", "task_status": "failed"}
 
    # 3. Primer paso determinista (sin investigación previa, ahorra tokens)
    if not research_data:
        print("🛡️ [Supervisor]: Sin investigación previa. Derivando a 'research_agent'.")
        return {"next_agent": "research_agent", "iteration_count": iteration_count + 1}
 
    # 4. Con análisis hecho, el cierre se decide en código (sin gastar tokens)
    if analysis_result:
        ok = analysis_result.get("status") == "success"
        print(f"🛡️ [Supervisor]: Análisis {'completado' if ok else 'fallido'}. Cierre.")
        return {"next_agent": "__end__", "task_status": "completed" if ok else "failed"}
 
    # 5. El LLM propone el siguiente paso
    llm = LLMFactory.create_llm(LLMFactory.get_default_config(provider=ClientType.GROQ))
    structured_llm = llm.with_structured_output(SupervisorOutput).with_retry(
        stop_after_attempt=2, wait_exponential_jitter=True
    )
    context = (
        f"Consulta del usuario: '{user_query}'\n"
        f"El usuario pidió un cálculo: {requires_calculation}"
    )
 
    try:
        response = await structured_llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)]
        )
        proposed = response.next_agent
        print(f"\n🧠 [Supervisor LLM Ruteo]: {proposed} | Razón: {response.reason}")
    except Exception as e:
        print(f"⚠️ [Supervisor Fallback]: Falló el LLM ({e}). Aplicando ruteo determinista.")
        proposed = "analyst_agent" if requires_calculation else "__end__"
 
    # 6. El LLM propone; el código valida
    if proposed == "research_agent":
        print("🛡️ [Supervisor]: La investigación ya está completa; no se repite.")
        proposed = "analyst_agent" if requires_calculation else "__end__"
    elif proposed == "analyst_agent" and not requires_calculation:
        print("🛡️ [Supervisor]: No se pidió ningún cálculo; no se deriva al analista.")
        proposed = "__end__"
    elif proposed == "__end__" and requires_calculation:
        print("🛡️ [Supervisor]: Se pidió un cálculo y falta el análisis; se deriva al analista.")
        proposed = "analyst_agent"
 
    result = {"next_agent": proposed, "iteration_count": iteration_count + 1}
    if proposed == "__end__":
        result["task_status"] = "completed"
    return result
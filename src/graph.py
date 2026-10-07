from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from Config import ClientType
from factory import LLMFactory
from src.tools import TOOLS

# Obtenemos la configuración por defecto configurada para GROQ
config_llm = LLMFactory.get_default_config(provider=ClientType.GROQ)

# Creamos el modelo a traves del Factory
llm = LLMFactory.create_llm(config_llm)
llm_with_tools = llm.bind_tools(TOOLS)

async def call_model(state: MessagesState):
    messages = state["messages"]
    system_prompt = SystemMessage(
        content="Eres un asistente de soporte técnico y gestión de clientes"
                "Responde con precisión apoyandote siempre en las herramientas disponibles"
    )
    response = await llm_with_tools.ainvoke([system_prompt] + messages)
    return {"messages": [response]}

# Construcción del grafo
builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(TOOLS))

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# Única fuente de verdad para obtener el ejecutor compilado
def get_agent_executor(checkpointer=None):
    return builder.compile(checkpointer=checkpointer)
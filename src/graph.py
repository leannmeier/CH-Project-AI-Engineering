from typing import Literal
from langgraph.graph import StateGraph, START, END
from state import MultiAgentState
from src.agents.supervisor import supervisor_node
from src.agents.research_agent import research_node
from src.agents.analyst_agent import analyst_node

def route_decision(state: MultiAgentState) -> Literal["research_agent", "analyst_agent", "__end__"]:
    task_status = state.get("task_status", "pending")
    if task_status == "failed":
        return "__end__"
        
    messages = state.get("messages", [])
    user_query = messages[0].content.lower() if messages else ""
    requires_calculation = any(k in user_query for k in ["calculá", "calcula", "promedio", "total", "suma", "métrica"])
    
    next_agent = state.get("next_agent", "__end__")

    if next_agent == "analyst_agent" and not requires_calculation:
        print("⚠️ [Guarda Grafo]: Intento ilegal de derivar al analista sin cálculos solicitados. Forzando cierre (__end__).")
        return "__end__"

    if next_agent in ["__end__", END]:
        return "__end__"

    return next_agent

def get_multi_agent_executor(checkpointer=None):
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", research_node)
    workflow.add_node("analyst_agent", analyst_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "research_agent": "research_agent",
            "analyst_agent": "analyst_agent",
            "__end__": END
        }
    )

    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("analyst_agent", "supervisor")

    return workflow.compile(checkpointer=checkpointer)
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import get_multi_agent_executor

async def run_agent(query: str, thread_id: str) -> str:
    """
    Ejecuta el sistema multiagente con Supervisor.

    Usa SQLite para persistir checkpoints del grafo.
    La API FastAPI utiliza su propio checkpointer en Redis.
    """

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 20,
    }

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "research_agent",
        "iteration_count": 0,
        "max_iterations": 6,
        "retry_count": 0,
        "research_data": [],
        "analysis_result": {},
        "task_status": "pending",
    }

    async with AsyncSqliteSaver.from_conn_string(
        "data/checkpoints.sqlite"
    ) as memory:

        agent_executor = get_multi_agent_executor(
            checkpointer=memory
        )

        result = await agent_executor.ainvoke(
            initial_state,
            config=config,
        )

        if result.get("task_status") == "failed":
            research = result.get("research_data") or []
            analysis = result.get("analysis_result") or {}

            error = (
                research[-1].get("error_message")
                if research and research[-1].get("status") == "error"
                else analysis.get("error_message")
            )

            return error or "La ejecución del agente falló."

        messages = result.get("messages", [])

        if not messages:
            return "El sistema no generó una respuesta."

        last_message = messages[-1]
        return str(getattr(last_message, "content", ""))

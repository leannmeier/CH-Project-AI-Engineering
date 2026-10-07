from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import get_agent_executor

async def run_agent(query: str, thread_id: str) -> str:
    """Ejecuta el agente para una consulta dada reutilizando el ejecutor centralizado"""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10
    }
    
    async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as memory:
        agent_executor = get_agent_executor(checkpointer=memory)
        input_data: MessagesState = {"messages": [HumanMessage(content=query)]}
        
        result = await agent_executor.ainvoke(input_data, config=config)
        
        last_message = result["messages"][-1]
        content = getattr(last_message, "content", "")
        return str(content)
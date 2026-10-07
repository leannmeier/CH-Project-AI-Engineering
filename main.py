import asyncio
import json
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import get_agent_executor

async def run_agent_workflow():
    async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as memory:
        # Reutilizamos la función centralizada de graph.py
        agent_executor = get_agent_executor(checkpointer=memory)

        config: RunnableConfig = {
            "configurable": {"thread_id": "session-user-102"},
            "recursion_limit": 10
        }

        print("Turno 1: Solicitud Multi-paso (ReAct + Tools)")
        query_1 = "Hola. ¿Cuántos pedidos completados tiene el usuario juan@mail.com y cuál es su monto total?"
        
        trace_logs = []

        async for event in agent_executor.astream_events(
            {"messages": [HumanMessage(content=query_1)]},
            config=config,
            version="v2"
        ):
            kind = event["event"]
            if kind == "on_chain_start":
                trace_logs.append({"event": "chain_start", "name": event.get("name")})
            elif kind == "on_tool_start":
                print(f"[Tool Call]: Executing {event['name']} with input: {event['data'].get('input')}")
                trace_logs.append({"event": "tool_start", "tool": event['name'], "input": event['data'].get('input')})
            elif kind == "on_tool_end":
                print(f"[Tool Output]: {event['data'].get('output')}")
                trace_logs.append({"event": "tool_end", "output": str(event['data'].get('output'))})

        with open("execution_trace.json", "w", encoding="utf-8") as f:
            json.dump(trace_logs, f, indent=2, ensure_ascii=False)
        print("\nExecution trace guardado en 'execution_trace.json'")

        print("\nTurno 2: Verificación de Persistencia (SqliteSaver)")
        query_2 = "¿Cuál era el nombre de ese cliente y el ID de su segundo pedido?"
        
        async for chunk in agent_executor.astream(
            {"messages": [HumanMessage(content=query_2)]},
            config=config
        ):
            if isinstance(chunk, dict):
                for node, values in chunk.items():
                    if isinstance(values, dict) and "messages" in values:
                        last_msg = values["messages"][-1]
                        if getattr(last_msg, "content", None):
                            print(f"Agent ({node}): {last_msg.content}")

if __name__ == "__main__":
    asyncio.run(run_agent_workflow())
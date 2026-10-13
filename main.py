import asyncio
import uuid
import argparse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import get_multi_agent_executor

async def main():
    parser = argparse.ArgumentParser(description="Ejecutor de Casos del Sistema Multi-Agent")
    parser.add_argument("--case", type=int, choices=[1, 2, 3], help="Número de caso específico a ejecutar (1, 2 o 3)")
    args = parser.parse_args()

    test_cases = {
        1: {"id": "CASO 1", "description": "Solo buscar/listar información sin cálculos", "query": "Busca información general sobre las últimas noticias de tecnología espacial."},
        2: {"id": "CASO 2", "description": "Reporte completo con búsqueda y cálculo uniforme", "query": "Buscá la población de Buenos Aires, Córdoba y Rosario y calculá el promedio."},
        3: {"id": "CASO 3", "description": "Consulta inexistente / Datos no hallados", "query": "Busca información sobre transacciones de un usuario totalmente inexistente llamado zzz999@mail.fake."}
    }

    cases_to_run = [args.case] if args.case else [1, 2, 3]

    async with AsyncSqliteSaver.from_conn_string("data/multi_agent_checkpoints.sqlite") as memory:
        agent_executor = get_multi_agent_executor(checkpointer=memory)

        for case_num in cases_to_run:
            case = test_cases[case_num]
            unique_thread_id = f"session-{uuid.uuid4()}"
            config: RunnableConfig = {"configurable": {"thread_id": unique_thread_id}, "recursion_limit": 15}

            print(f"\n" + "="*60)
            print(f"🚀 EJECUTANDO {case['id']}: {case['description']}")
            print(f"👤 Query: {case['query']}")
            print("="*60)

            initial_state = {
                "messages": [HumanMessage(content=case["query"])],
                "next_agent": "research_agent",
                "iteration_count": 0,
                "max_iterations": 6,
                "retry_count": 0,
                "research_data": [],
                "analysis_result": {},
                "task_status": "pending"
            }

            try:
                async for chunk in agent_executor.astream(initial_state, config=config):
                    for node_name, node_output in chunk.items():
                        print(f"🔄 [Nodo Ejecutado]: {node_name}")
                        if "next_agent" in node_output:
                            print(f"   👉 Siguiente paso: {node_output['next_agent']}")
                        print("-" * 40)

                final_state = await agent_executor.aget_state(config)
                values = final_state.values
                task_status = values.get("task_status", "pending")
                
                print(f"\n🏁 Estado final ({case['id']}) - task_status: {task_status}")
                print(f"📦 Conteo total de mensajes acumulados: {len(values.get('messages', []))}")
                
                if task_status == "failed":
                    err = "La tarea no pudo completarse debido a un fallo en los agentes."
                    if values.get("research_data") and values["research_data"][-1].get("error_message"):
                        err = values["research_data"][-1].get("error_message")
                    elif values.get("analysis_result") and values["analysis_result"].get("error_message"):
                        err = values["analysis_result"].get("error_message")
                    print(f"❌ Respuesta del Sistema (Fallo controlado): {err}")
                else:
                    messages = values.get("messages", [])
                    if messages:
                        print(f"🤖 Respuesta:\n{messages[-1].content}")
                    analysis = values.get("analysis_result", {})
                    if analysis and analysis.get("metrics"):
                        print(f"\n📊 Métricas Calculadas:\n{analysis.get('metrics')}")
                print("="*60)

            except Exception as e:
                print(f"❌ Error crítico en {case['id']}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
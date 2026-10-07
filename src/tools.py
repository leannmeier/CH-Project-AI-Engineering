from typing import Dict, Any
from langchain_core.tools import tool
# Importamos directamente desde la raíz
from rag_chain import build_hybrid_retriever 

MOCK_USERS_DB = {
    "juan@mail.com": {"cliente_id": "102", "nombre": "Juan Perez"},
    "maria@gmail.com": {"cliente_id": "105", "nombre": "Maria Gómez"}
}

MOCK_ORDERS_DB = {
    "102": [
        {"id_pedido": "P-001", "monto": 4500, "estado": "Completado"},
        {"id_pedido": "P-002", "monto": 10000, "estado": "Completado"},
        {"id_pedido": "P-003", "monto": 2500, "estado": "Cancelado"}
    ]
}

@tool
async def consultar_base_conocimiento(query: str) -> str:
    """
    Consulta la base de conocimiento técnica (Pinecone + BM25) para obtener 
    información detallada sobre documentación o procedimientos.
    """
    retriever = build_hybrid_retriever()
    docs = await retriever.ainvoke(query)
    if not docs:
        return "No se encontró información relevante en la base de conocimiento."
    return "\n\n".join([doc.page_content for doc in docs])

@tool
async def buscar_cliente_por_email(email: str) -> Dict[str, Any]:
    """Busca la información básica de un cliente por su email para obtener cliente_id."""
    cliente = MOCK_USERS_DB.get(email.lower().strip())
    if not cliente:
        return {"error": f"No se encontró cliente con el email: {email}"}
    return cliente

@tool
async def obtener_pedidos_cliente(cliente_id: str) -> Dict[str, Any]:
    """Obtiene los pedidos asociados a un cliente_id."""
    pedidos = MOCK_ORDERS_DB.get(cliente_id)
    if not pedidos:
        return {"error": f"No se encontraron pedidos para el cliente_id: {cliente_id}"}
    
    total_acumulado = sum(p["monto"] for p in pedidos if p["estado"] == "Completado")
    return {
        "cliente_id": cliente_id,
        "cantidad_pedidos": len(pedidos),
        "total_completado": total_acumulado,
        "pedidos": pedidos
    }

TOOLS = [buscar_cliente_por_email, obtener_pedidos_cliente, consultar_base_conocimiento]
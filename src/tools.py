from langchain_tavily import TavilySearch
from langchain_core.tools import tool

tavily_tool = TavilySearch(max_results=3) # Subido a 3 para capturar mejor las múltiples entidades

@tool
def calculate_metrics(data_list: list) -> dict:
    """
    Calcula métricas agregadas (total, promedio, máximo y mínimo) 
    a partir de una lista de valores numéricos.
    """
    if not data_list:
        return {"error": "No hay datos numéricos para calcular."}
    
    total = sum(data_list)
    count = len(data_list)
    average = total / count if count > 0 else 0
    maximum = max(data_list)
    minimum = min(data_list)
    
    return {
        "total": total,
        "count": count,
        "average": average,
        "max": maximum,
        "min": minimum
    }

RESEARCH_TOOLS = [tavily_tool]
ANALYST_TOOLS = [calculate_metrics]
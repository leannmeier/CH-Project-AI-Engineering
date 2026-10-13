from typing import TypedDict, Annotated, List, Dict, Any
import operator

class MultiAgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    next_agent: str
    iteration_count: int
    max_iterations: int
    retry_count: int
    research_data: Annotated[List[Dict[str, Any]], operator.add]
    analysis_result: Dict[str, Any]
    task_status: str
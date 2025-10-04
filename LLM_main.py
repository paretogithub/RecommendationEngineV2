from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph,END, START

from condition_agents.tactic_category_nodes import nutrition_node,exercise_node,lifestyle_node,supplement_node

from condition_agents.patient_data_read import patient_data_node
from condition_agents.cross_validator_node import cross_validator_agent_node
from condition_agents.final_generator_node_structured import final_generator_agent_node 

class AgentState(TypedDict, total=False):
    patient_data: Any
    nutrition_output: Any
    exercise_output: Any
    lifestyle_output: Any
    supplement_output: Any
    cross_validated: Optional[str]
    # Final recommendation result
    final_output: Optional[dict]


def build_agent_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("patient_data_node", patient_data_node)
    graph.add_node("nutrition_agent_node", nutrition_node)
    graph.add_node("exercise_agent_node", exercise_node)
    graph.add_node("lifestyle_agent_node", lifestyle_node)
    graph.add_node("supplement_agent_node", supplement_node)
    graph.add_node("cross_validator_node", cross_validator_agent_node)
    graph.add_node("final_generator_node", final_generator_agent_node)

    # patient_data_node runs once at start (to save patient data in state)
    graph.add_edge("__start__", "patient_data_node")

    # fan-out independently for uploads
    graph.add_edge("__start__", "nutrition_agent_node")
    graph.add_edge("__start__", "exercise_agent_node")
    graph.add_edge("__start__", "lifestyle_agent_node")
    graph.add_edge("__start__", "supplement_agent_node")

    # merge into cross-validator
    graph.add_edge("nutrition_agent_node", "cross_validator_node")
    graph.add_edge("exercise_agent_node", "cross_validator_node")
    graph.add_edge("lifestyle_agent_node", "cross_validator_node")
    graph.add_edge("supplement_agent_node", "cross_validator_node")

    # validator → final generator → END
    graph.add_edge("cross_validator_node", "final_generator_node")
    graph.add_edge("final_generator_node", END)

    return graph.compile()


agent_graph = build_agent_graph()

####

if __name__ == "__main__":
    #graph = build_agent_graph()
    initial_state = {}
    final_state = agent_graph.invoke(initial_state)

    # print("✅ Final Validated Recommendation Output:\n")
    # print(final_state["final_output"])


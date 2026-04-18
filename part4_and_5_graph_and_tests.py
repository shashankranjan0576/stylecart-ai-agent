from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from part2_and_3_state_and_nodes import *

def route_decision(state):
    return state["route"]

def eval_decision(state):
    return "save"

graph = StateGraph(CapstoneState)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")

graph.add_edge("memory","router")
graph.add_conditional_edges("router", route_decision,{
    "retrieve":"retrieve",
    "tool":"tool"
})

graph.add_edge("retrieve","answer")
graph.add_edge("tool","answer")
graph.add_edge("answer","eval")
graph.add_edge("eval","save")
graph.add_edge("save", END)

app = graph.compile(checkpointer=MemorySaver())

def ask(q, thread_id="1"):
    return app.invoke({"question":q}, config={"configurable":{"thread_id":thread_id}})

# TEST
if __name__ == "__main__":
    print(ask("What is return policy?"))
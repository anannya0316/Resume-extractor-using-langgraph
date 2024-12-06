from langgraph.graph import StateGraph, START, END

builder = StateGraph({})
builder.add_node("start", lambda state: state)
builder.add_edge(START, "start")
graph = builder.compile()
print(graph)

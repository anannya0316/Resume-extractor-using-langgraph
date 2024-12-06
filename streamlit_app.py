from langgraph.graph import StateGraph, START, END

state_schema = {
    "pdf_path": str,
    "raw_text": str,
    "formatted_text": str,
    "education": str,
    "experience": str,
    "skills": str,
    "summary": str,
    "sections": dict,
    "selected_model": str,
    "memory_log": list,
}

builder = StateGraph(state_schema)
builder.add_node("start", lambda state: state)
builder.add_edge(START, "start")
graph = builder.compile()
print(graph)

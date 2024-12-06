import streamlit as st
from langgraph.graph import StateGraph, START, END

# Define a simple state schema
state_schema = {
    "input_text": str,
    "output_text": str,
}

# Initialize the graph
builder = StateGraph(state_schema)

# Define nodes
def input_node(state):
    """Simulates an input node."""
    return {"input_text": state["input_text"]}

def processing_node(state):
    """Simulates a processing node."""
    processed_text = f"Processed: {state['input_text']}"
    return {"output_text": processed_text}

# Add nodes to the graph
builder.add_node("input", input_node)
builder.add_node("process", processing_node)

# Add edges
builder.add_edge(START, "input")
builder.add_edge("input", "process")
builder.add_edge("process", END)

# Compile the graph
graph = builder.compile()

# Streamlit app
st.title("Langgraph Test in Streamlit")

# Input from user
user_input = st.text_input("Enter some text", "")

# Run the graph if input is provided
if user_input:
    # Initial state for the graph
    initial_state = {
        "input_text": user_input,
        "output_text": "",
    }

    # Execute the graph
    result = graph.invoke(initial_state)

    # Display results
    st.subheader("Graph Execution Result")
    st.write("Input Text:", result["input_text"])
    st.write("Output Text:", result["output_text"])

import os
from pymongo import MongoClient
import getpass
import PyPDF2
import openai
from langdetect import detect
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Optional, Annotated
# from IPython.display import Image
import json
import toml

# Load the configuration file
config = toml.load("config.toml")

# Access the keys
langchain_api_key = config["api_keys"]["langchain"]
openai_api_key = config["api_keys"]["openai"]
tavily_api_key = config["api_keys"]["tavily"]
mongo_uri = config["database"]["mongo_uri"]
langchain_tracing_v2 = config["tracing"]["langchain_v2"]

client = MongoClient(mongo_uri)
db = client["Resume_extraction"]
collection = db["extraction_logs"]

# PromptManager class to manage multiple prompts
class PromptManager:
    def __init__(self):
        self.prompts = {
            "formatting": [
                "You are a professional document formatter. Structure the following raw text from a resume logically into sections in JSON format: "
                "Use sections like Contact Information, Skills, Education, Work Experience, Projects, Publications, Certificates, and Organizations if applicable.",
                
                "As an expert document formatter, your task is to convert the following raw text from a resume into a structured JSON format. "
                "The JSON should include sections such as Contact Information, Skills, Education, Work Experience, Projects, and other relevant headers.",
                
                "Act as a document formatting specialist and transform the given raw resume text into a clean JSON format. "
                "Organize it into sections like Contact Information, Skills, Education, Work Experience, Projects, Publications, Certificates, and Organizations."
            ]
        }

    def get_prompts(self, task: str):
        return self.prompts.get(task, [])

# State definition with sections reducer
def resume_sections_reducer(left, right):
    if not isinstance(left, dict):
        left = {"sections": [left]} if left else {"sections": []}
    if not isinstance(right, dict):
        right = {"sections": [right]} if right else {"sections": []}
    return {
        "pdf_path": left.get("pdf_path") or right.get("pdf_path"),
        "raw_text": left.get("raw_text") or right.get("raw_text"),
        "formatted_text": left.get("formatted_text") or right.get("formatted_text"),
        "education": left.get("education") or right.get("education"),
        "experience": left.get("experience") or right.get("experience"),
        "skills": left.get("skills") or right.get("skills"),
        "summary": left.get("summary") or right.get("summary"),
        "sections": left.get("sections", []) + right.get("sections", []),
        "selected_model": left.get("selected_model") or right.get("selected_model"),
        "memory_log": left.get("memory_log") or right.get("memory_log")
    }

class State(TypedDict):
    pdf_path: str
    raw_text: Optional[str]
    formatted_text: Optional[str]
    education: Optional[str]
    experience: Optional[str]
    skills: Optional[str]
    summary: Optional[str]
    sections: Annotated[dict, resume_sections_reducer]
    selected_model: Optional[str]
    memory_log: list

# Utility functions
def detect_format(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return "pdf"
    elif file_path.endswith(".jpg") or file_path.endswith(".png"):
        return "image"
    else:
        return "text"

def analyze_content(input_text: str) -> str:
    return "complex" if len(input_text.split()) > 500 else "simple"

def route_llm(input_text: str, file_path: str) -> str:
    format_type = detect_format(file_path)
    content_type = analyze_content(input_text)
    language = detect(input_text)

    if format_type == "image" or content_type == "complex" or language != "en":
        return "gpt-4"
    return "gpt-3.5-turbo"

def call_with_retry(prompt, model, retries=3):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return json.loads(response['choices'][0]['message']['content'])
        except json.JSONDecodeError:
            prompt = f"Ensure the response is in valid JSON format. {prompt}"
        except Exception as e:
            print(f"Retry {attempt + 1} failed: {e}")
    return {"error": "Failed to get structured JSON after retries"}

# Logging function
def log_to_mongo(data: dict):
    collection.insert_one(data)

# Graph Node Functions
def input_node(state: State) -> dict:
    return state

def extraction_node(state: State) -> dict:
    with open(state["pdf_path"], 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        raw_text = "".join(page.extract_text() for page in reader.pages)
    log_to_mongo({"stage": "extraction", "raw_text": raw_text})
    return {"raw_text": raw_text}

def routing_node(state: State) -> dict:
    selected_model = route_llm(state["raw_text"], state["pdf_path"])
    log_to_mongo({"stage": "routing", "selected_model": selected_model})
    return {"selected_model": selected_model}

def formatting_node(state: State) -> dict:
    prompt_manager = PromptManager()
    prompts = prompt_manager.get_prompts("formatting")
    results = []
    
    for prompt in prompts:
        try:
            response = openai.ChatCompletion.create(
                model=state["selected_model"],
                messages=[{"role": "user", "content": f"{prompt} {state['raw_text']}"}],
                temperature=0
            )
            results.append(response['choices'][0]['message']['content'])
        except Exception as e:
            results.append(None)
    
    log_to_mongo({"stage": "formatting", "prompts": prompts, "results": results})
    return {"formatted_text": results[0]} if results[0] else {"formatted_text": None}

def education_node(state: State) -> dict:
    prompt = f"Extract education details:\n{state['formatted_text']}"
    response = call_with_retry(prompt, state["selected_model"])
    log_to_mongo({"stage": "education_extraction", "education": response})
    return {"education": response}

def experience_node(state: State) -> dict:
    prompt = f"Extract work experience details:\n{state['formatted_text']}"
    response = call_with_retry(prompt, state["selected_model"])
    log_to_mongo({"stage": "experience_extraction", "experience": response})
    return {"experience": response}

def skills_node(state: State) -> dict:
    prompt = f"Extract skills details:\n{state['formatted_text']}"
    response = call_with_retry(prompt, state["selected_model"])
    log_to_mongo({"stage": "skills_extraction", "skills": response})
    return {"skills": response}

def summary_node(state: State) -> dict:
    summary_prompt = (
        f"Summarize the following sections:\n\n"
        f"Education: {state['education']}\n"
        f"Experience: {state['experience']}\n"
        f"Skills: {state['skills']}\n"
    )
    response = call_with_retry(summary_prompt, state["selected_model"])
    log_to_mongo({"stage": "summary_generation", "summary": response})
    return {"summary": response}

def human_feedback_node(state: State) -> dict:
    feedback = input("Provide feedback (or type 'good' if satisfied): ")
    log_to_mongo({"stage": "feedback", "feedback": feedback})
    if feedback.lower() == "good":
        return state

    refined_prompt = f"Refine output based on this feedback:\n{feedback}\n\nOriginal Output:\n{state['formatted_text']}"
    response = call_with_retry(refined_prompt, state["selected_model"])
    log_to_mongo({"stage": "refinement", "refined_output": response})
    return {"formatted_text": response}

# Graph Construction
builder = StateGraph(State)
builder.add_node("input", input_node)
builder.add_node("text_extraction", extraction_node)
builder.add_node("routing", routing_node)
builder.add_node("formatting", formatting_node)
builder.add_node("education_extractor", education_node)
builder.add_node("experience_extractor", experience_node)
builder.add_node("skills_extractor", skills_node)
builder.add_node("resume_summary", summary_node)
builder.add_node("human_feedback", human_feedback_node)

builder.add_edge(START, "input")
builder.add_edge("input", "text_extraction")
builder.add_edge("text_extraction", "routing")
builder.add_edge("routing", "formatting")
builder.add_edge("formatting", "education_extractor")
builder.add_edge("formatting", "experience_extractor")
builder.add_edge("formatting", "skills_extractor")
builder.add_edge("education_extractor", "resume_summary")
builder.add_edge("experience_extractor", "resume_summary")
builder.add_edge("skills_extractor", "resume_summary")
builder.add_edge("resume_summary", "human_feedback")
builder.add_edge("human_feedback", END)

graph = builder.compile()

# Initial State
initial_state = {
    "pdf_path": "example_resume.pdf",
    "raw_text": None,
    "formatted_text": None,
    "education": None,
    "experience": None,
    "skills": None,
    "summary": None,
    "sections": {},
    "selected_model": None,
    "memory_log": []
}

# Execute the Graph
result = graph.invoke(initial_state)
print("Final Output:", result)

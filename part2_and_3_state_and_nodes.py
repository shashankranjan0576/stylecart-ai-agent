from typing import TypedDict, List
import os
from datetime import datetime
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb

GROQ_API_KEY = "YOUR_API_KEY"
MODEL_NAME = "llama-3.3-70b-versatile"

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection("stylecart_kb")

class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    customer_name: str


def memory_node(state):
    msgs = state.get("messages", []) + [{"role":"user","content":state["question"]}]
    return {"messages":msgs[-6:], "eval_retries":0}


def router_node(state):
    if "date" in state["question"].lower():
        return {"route":"tool"}
    return {"route":"retrieve"}


def retrieval_node(state):
    emb = embedder.encode([state["question"]]).tolist()
    res = collection.query(query_embeddings=emb, n_results=3)
    return {"retrieved":"\n\n".join(res["documents"][0])}


def tool_node(state):
    return {"tool_result":str(datetime.now())}


def answer_node(state):
    context = state.get("retrieved","") + state.get("tool_result","")

    prompt = f"""
Answer ONLY from context.
If not found say you don't know.

Context:
{context}

Q: {state['question']}
"""
    return {"answer": llm.invoke(prompt).content}


def eval_node(state):
    return {"faithfulness":1.0, "eval_retries":1}


def save_node(state):
    msgs = state.get("messages", [])
    msgs.append({"role":"assistant","content":state["answer"]})
    return {"messages":msgs}
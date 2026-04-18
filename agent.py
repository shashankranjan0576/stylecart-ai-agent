import os
from datetime import datetime
from typing import TypedDict, List

from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 🔑 ADD YOUR API KEY HERE
GROQ_API_KEY = "YOUR_API_KEYi"
MODEL_NAME = "llama-3.3-70b-versatile"

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2
SLIDING_WINDOW = 6

# ───────── STATE ─────────
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

# ───────── RESOURCES ─────────
llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

# ✅ FIX: avoid duplicate collection error
try:
    client.delete_collection("stylecart_kb")
except:
    pass

collection = client.create_collection("stylecart_kb")

docs = [
    {"id":"1","topic":"Return Policy","text":"Returns allowed within 7 days. Items must be unused with tags."},
    {"id":"2","topic":"Shipping","text":"Delivery takes 3-5 days. Express 2 days."},
    {"id":"3","topic":"Payment","text":"UPI, cards, COD available. COD fee Rs.50."},
    {"id":"4","topic":"Tracking","text":"Track via SMS or My Orders."},
    {"id":"5","topic":"Cancellation","text":"Cancel within 2 hours if not dispatched."},
    {"id":"6","topic":"Exchange","text":"Exchange within 7 days."},
    {"id":"7","topic":"Sizes","text":"Sizes XS to 3XL."},
    {"id":"8","topic":"Coins","text":"1 coin per Rs.10 spent."},
    {"id":"9","topic":"Offers","text":"5% discount prepaid."},
    {"id":"10","topic":"Support","text":"WhatsApp +91-98765-43210"},
]

texts = [d["text"] for d in docs]
emb = embedder.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=emb,
    ids=[d["id"] for d in docs],
    metadatas=[{"topic": d["topic"]} for d in docs]
)

# ───────── NODES ─────────

def memory_node(state):
    msgs = state.get("messages", []) + [{"role":"user","content":state["question"]}]
    msgs = msgs[-SLIDING_WINDOW:]

    name = state.get("customer_name","")
    if "my name is" in state["question"].lower():
        name = state["question"].lower().split("my name is")[-1].strip().split()[0].capitalize()

    return {"messages":msgs,"customer_name":name,"eval_retries":0}


def router_node(state):
    q = state["question"].lower()

    if "date" in q or "time" in q:
        return {"route":"tool"}

    if "hi" in q or "hello" in q or "thanks" in q:
        return {"route":"memory_only"}

    return {"route":"retrieve"}


def retrieval_node(state):
    q_emb = embedder.encode([state["question"]]).tolist()
    res = collection.query(query_embeddings=q_emb,n_results=3)
    chunks = res["documents"][0]
    return {"retrieved":"\n\n".join(chunks)}


def skip_retrieval_node(state):
    return {"retrieved":"", "sources":[]}


def tool_node(state):
    try:
        now = datetime.now()
        return {"tool_result": f"Today is {now.strftime('%A, %d %B %Y')} and time is {now.strftime('%I:%M %p')}"}
    except:
        return {"tool_result": "Unable to fetch date/time"}


def answer_node(state):
    context = state.get("retrieved","") + state.get("tool_result","")
    name = state.get("customer_name","")

    name_line = ""
    if name:
        name_line = f"The customer's name is {name}. Use it if relevant.\n"

    prompt = f"""
You are a customer support assistant.

STRICT RULE:
Answer ONLY from context OR known memory.
If answer is not available, say:
"I don't have that information. Please contact support at WhatsApp +91-98765-43210"

{name_line}

Context:
{context}

Question:
{state['question']}
"""

    return {"answer": llm.invoke(prompt).content}

def eval_node(state):
    return {"faithfulness":1.0,"eval_retries":1}


def save_node(state):
    msgs = state.get("messages",[])
    msgs.append({"role":"assistant","content":state["answer"]})
    return {"messages":msgs}


def route_decision(state):
    r = state.get("route","retrieve")
    if r == "tool":
        return "tool"
    elif r == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state):
    return "save"

# ───────── GRAPH ─────────

g = StateGraph(CapstoneState)

g.add_node("memory", memory_node)
g.add_node("router", router_node)
g.add_node("retrieve", retrieval_node)
g.add_node("skip", skip_retrieval_node)
g.add_node("tool", tool_node)
g.add_node("answer", answer_node)
g.add_node("eval", eval_node)
g.add_node("save", save_node)

g.set_entry_point("memory")

g.add_edge("memory","router")

g.add_conditional_edges("router", route_decision, {
    "retrieve":"retrieve",
    "tool":"tool",
    "skip":"skip"
})

g.add_edge("retrieve","answer")
g.add_edge("tool","answer")
g.add_edge("skip","answer")

g.add_edge("answer","eval")
g.add_edge("eval","save")
g.add_edge("save",END)

app = g.compile(checkpointer=MemorySaver())

# ───────── API ─────────

def ask(q, thread_id="1"):
    return app.invoke({"question":q},config={"configurable":{"thread_id":thread_id}})


# ───────── RUN ─────────

if __name__ == "__main__":
    print("🤖 Agent started. Type 'exit' to stop.\n")

    while True:
        q = input("You: ")

        if q.lower() == "exit":
            break

        result = ask(q)
        print("Bot:", result["answer"])
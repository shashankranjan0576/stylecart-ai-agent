from sentence_transformers import SentenceTransformer
import chromadb

print("Loading model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    {"id":"1","topic":"Return Policy","text":"Returns allowed within 7 days. Items must be unused with tags."},
    {"id":"2","topic":"Shipping","text":"Delivery takes 3-5 days. Express delivery 2 days."},
    {"id":"3","topic":"Payment","text":"UPI, cards, COD available. COD fee Rs.50."},
    {"id":"4","topic":"Tracking","text":"Track via SMS or My Orders."},
    {"id":"5","topic":"Exchange","text":"Exchange within 7 days."},
    {"id":"6","topic":"Cancellation","text":"Cancel within 2 hours if not dispatched."},
    {"id":"7","topic":"Sizes","text":"Sizes XS to 3XL."},
    {"id":"8","topic":"Coins","text":"1 coin per Rs.10 spent."},
    {"id":"9","topic":"Offers","text":"5% discount prepaid."},
    {"id":"10","topic":"Support","text":"WhatsApp +91-98765-43210"}
]

client = chromadb.Client()
collection = client.create_collection("stylecart_kb")

texts = [d["text"] for d in documents]
embeddings = embedder.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[d["id"] for d in documents],
    metadatas=[{"topic": d["topic"]} for d in documents]
)

print("✅ KB Ready")

# TEST
query = "return policy"
res = collection.query(query_embeddings=embedder.encode([query]).tolist(), n_results=2)
print(res["documents"])
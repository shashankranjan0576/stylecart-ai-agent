from agent import ask

qs = [
    "What is return policy?",
    "How long delivery takes?",
    "Do you have COD?",
    "How to cancel order?",
    "What is todays date?"
]

for q in qs:
    print("\nQ:",q)
    print("A:",ask(q)["answer"])
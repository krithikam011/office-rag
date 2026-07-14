"""
chat.py
-------
Simple command-line chat loop for the Office Policy Assistant.

Run this after ingest.py has built the vector store:
    python chat.py
"""

from rag_engine import answer_question


def main():
    print("Office Policy Assistant (type 'exit' to quit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not query:
            continue

        try:
            answer = answer_question(query)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

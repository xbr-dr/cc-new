import re
import ollama
from rag_retriever import retrieve_relevant_chunks

GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

def generate_answer(history):
    if not history or not isinstance(history, list):
        return "Invalid chat history."

    last_message = history[-1]["content"].strip().lower()

    # Quick response for greetings (no RAG, no model call)
    if last_message in GREETINGS:
        return "Hello! How can I assist you with campus or college-related questions today?"

    # Retrieve RAG context for the latest user question
    relevant_docs = retrieve_relevant_chunks(last_message, top_k=5)
    context_text = "\n\n".join(relevant_docs).strip()

    # System prompts — refined
    system_prompts = [
        {
            "role": "system",
            "content": (
                "You are CampusGPT, a polite, professional assistant that ONLY answers "
                "questions related to our college, campus facilities, departments, staff, events, "
                "courses, admission process, and official guidelines.\n\n"
                "Check the provided context for every question. "
                "If relevant information is found, use it to answer accurately and concisely.\n\n"
                "If the current question is unrelated to the campus or has no matching information "
                "in the context, respond exactly:\n"
                "'I could not find that information in our campus resources. Please rephrase your question so it relates to the campus.'\n\n"
                "Do not guess or make up details. Do not refuse future relevant questions because of past irrelevant ones.\n"
                "Never include <think> tags, reasoning steps, or internal thoughts in your answer."
            )
        },
        {"role": "system", "content": f"Context:\n{context_text if context_text else '(No relevant campus info found)'}"}
    ]

    # Combine with full conversation history
    messages = system_prompts + history

    # Call Ollama
    response = ollama.chat(
        model="qwen3:0.6b",  # ensure model is installed
        messages=messages,
        options={"temperature": 0.3}
    )

    # Clean unwanted <think> tags if they appear
    answer = response["message"]["content"]
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    return answer

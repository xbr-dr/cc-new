"""
Lightweight RAG Generator Module
--------------------------------
Uses Ollama with a small local model (e.g., phi3:mini)
or can be easily switched to an API-based model like OpenAI, Groq, or Gemini.

Functions:
- generate_answer(history)
"""

import re
from rag_retriever import retrieve_relevant_chunks

# Optional: import ollama if installed
try:
    import ollama
    _USE_OLLAMA = True
except ImportError:
    _USE_OLLAMA = False
    import requests  # fallback for API option


# === CONFIGURATION ===
SMALL_MODEL = "qwen3:0.6b"  # small, fast Ollama model
API_FALLBACK_URL = ""       # add endpoint if using external API (optional)


def generate_answer(history):
    """Generates a context-aware answer from chat history."""
    if not history or not isinstance(history, list):
        return "Invalid chat history."

    # Get latest user message
    user_message = history[-1]["content"] if history[-1]["role"] == "user" else ""
    if not user_message.strip():
        return "Please ask a valid question."

    # Retrieve relevant chunks from knowledge base
    relevant_docs = retrieve_relevant_chunks(user_message, top_k=5)
    context_text = "\n\n".join(relevant_docs) if relevant_docs else "No context found."

    # Strict system behavior (campus-related focus)
    system_instructions = """
You are CampusGPT, a helpful assistant for a college campus.
Answer ONLY questions related to the campus, its facilities, staff, departments, courses, events, contact details, and other official information.
If the question is unrelated to campus topics, politely decline to answer.
"""

    # Combine everything into a structured prompt
    messages = [
        {"role": "system", "content": system_instructions.strip()},
        {"role": "system", "content": f"Relevant context from documents:\n{context_text}"},
    ] + history

    try:
        # === Option 1: Local Ollama ===
        if _USE_OLLAMA:
            response = ollama.chat(
                model=SMALL_MODEL,
                messages=messages,
                options={"temperature": 0.6, "num_predict": 250}
            )
            assistant_reply = response["message"]["content"]

        # === Option 2: Remote API (if you add one) ===
        elif API_FALLBACK_URL:
            payload = {"messages": messages}
            res = requests.post(API_FALLBACK_URL, json=payload, timeout=30)
            assistant_reply = res.json().get("reply", "Error: No response from API.")
        else:
            return "Error: No model or API configured. Please install Ollama or set API_FALLBACK_URL."

        # Clean output (remove <think> tags etc.)
        assistant_reply = re.sub(r"<think>.*?</think>", "", assistant_reply, flags=re.DOTALL).strip()

    except Exception as e:
        assistant_reply = f"Error generating answer: {e}"

    return assistant_reply

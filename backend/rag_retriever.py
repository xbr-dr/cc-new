import os
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

nltk.download('punkt_tab')

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

corpus = []
index = None

def load_documents_and_build_index(doc_folder='knowledge_base/docs'):
    global corpus, index
    all_text = ""

    if not os.path.exists(doc_folder):
        print(f"Document folder '{doc_folder}' does not exist. Skipping index build.")
        return

    for filename in os.listdir(doc_folder):
        filepath = os.path.join(doc_folder, filename)
        if filename.lower().endswith(".pdf"):
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            all_text += text + "\n"
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        elif filename.lower().endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(filepath)
                text = df.to_string(index=False)
                if text:
                    all_text += text + "\n"
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        # Extend to other formats if needed

    if not all_text.strip():
        print("No text extracted from documents.")
        corpus = []
        index = None
        return

    sentences = sent_tokenize(all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    corpus = sentences

    corpus_embeddings = embed_model.encode(corpus, show_progress_bar=True)
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(corpus_embeddings).astype("float32"))

    print(f"Loaded {len(corpus)} text chunks into FAISS index.")

def retrieve_relevant_chunks(query, top_k=5):
    global index, corpus
    if index is None or len(corpus) == 0:
        return []

    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)
    return [corpus[i] for i in indices[0] if i < len(corpus)]

def clear_index():
    global corpus, index
    corpus = []
    index = None

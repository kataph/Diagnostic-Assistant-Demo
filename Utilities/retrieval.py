import os
import pickle
import hashlib
from typing import TypedDict
import numpy as np
import tiktoken
from openai import OpenAI

# =====================
# CONFIG
# =====================

CACHE_PATH = "embeddings_cache.pkl"

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4.1"
TOKENIZER_MODEL = "cl100k_base"

CHUNK_SIZE = 400        # tokens
CHUNK_OVERLAP = 0      # tokens
TOP_K = 2

# =====================
# UTILITIES
# =====================


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =====================
# CHUNKING
# =====================

def chunk_text(text, tokenizer, chunk_size, overlap) -> list[str]:
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


class Chunk(TypedDict):
    text: str
    source: str
    hash: str


def load_and_chunk_documents(folder_path, chunk_size, chunk_overlap, tokenizer_model) -> list[Chunk]:
    chunks = []

    tokenizer = tiktoken.get_encoding(tokenizer_model)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        text_chunks = chunk_text(text, tokenizer, chunk_size, chunk_overlap)

        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "source": filename,
                "hash": hash_text(chunk)
            })

    return chunks


# =====================
# EMBEDDING CACHE
# =====================

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache, cache_path):
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def _make_cache_key(chunk_hash: str, embed_model: str) -> str:
    """Cache key that binds a chunk hash to its embedding model."""
    return f"{embed_model}::{chunk_hash}"


def embed_texts(texts, client: OpenAI, embed_model: str):
    response = client.embeddings.create(
        model=embed_model,
        input=texts
    )
    return [e.embedding for e in response.data]


# =====================
# BUILD / LOAD INDEX
# =====================

# In-process index memo: avoids re-reading disk on every query within the same run.
_index_memo: dict[tuple, tuple] = {}


def get_chunks_and_embeddings(client: OpenAI, folder_path, chunk_size, chunk_overlap, tokenizer_model, embed_model, cache_path):
    key = (folder_path, chunk_size, chunk_overlap, tokenizer_model, embed_model)
    if key in _index_memo:
        return _index_memo[key]

    print("Loading documents...")
    chunks = load_and_chunk_documents(
        folder_path, chunk_size, chunk_overlap, tokenizer_model)

    cache = load_cache(cache_path)

    new_texts = []
    new_hashes = []

    for c in chunks:
        if _make_cache_key(c["hash"], embed_model) not in cache:
            new_texts.append(c["text"])
            new_hashes.append(c["hash"])

    if new_texts:
        print(f"Embedding {len(new_texts)} new chunks...")
        new_embeddings = embed_texts(new_texts, client, embed_model)

        for h, emb in zip(new_hashes, new_embeddings):
            cache[_make_cache_key(h, embed_model)] = np.array(emb)

        save_cache(cache, cache_path)
    else:
        print("No new chunks to embed.")

    chunk_embeddings = np.array([cache[_make_cache_key(c["hash"], embed_model)] for c in chunks])

    print(f"Index ready ({len(chunks)} chunks).")

    _index_memo[key] = (chunks, chunk_embeddings)
    return chunks, chunk_embeddings

# =====================
# RETRIEVAL
# =====================


def retrieve_top_chunks(query: str, client: OpenAI, top_k: int = TOP_K, folder_path: str | None = None, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, tokenizer_model=TOKENIZER_MODEL, embed_model=EMBED_MODEL, cache_path=CACHE_PATH) -> list[Chunk]:
    if folder_path is None:
        raise ValueError("folder_path must be provided (no default path is configured)")
    chunks, chunk_embeddings = get_chunks_and_embeddings(
        client, folder_path, chunk_size, chunk_overlap, tokenizer_model, embed_model, cache_path)
    query_embedding = embed_texts([query], client, embed_model)[0]

    scores = [
        cosine_similarity(query_embedding, emb)
        for emb in chunk_embeddings
    ]

    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


# =====================
# GENERATION (Responses API)
# =====================

def answer_question(question, client: OpenAI):
    retrieved = retrieve_top_chunks(question, client)

    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in retrieved
    )
    print(f"{' Found the following context: ':*^120}\n{context}\n"+"*"*120)

    response = client.responses.create(
        model=GEN_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY using the provided context. "
                    "If the answer is not present, say you don't know."
                )
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ]
    )

    return response.output_text


# =====================
# INTERACTIVE LOOP
# =====================

if __name__ == "__main__":
    client = OpenAI()
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break

        print("\nAnswer:\n")
        print(answer_question(q, client))

#!/usr/bin/env python3
"""
txt_to_chunks_csv.py (updated for openai>=1.0.0)

Function:
 - Read all .txt files under --input_dir
 - Chunk each file (word-based, configurable)
 - Call OpenAI embeddings (OpenAI v1 python client)
 - Write results to a CSV with columns:
    id, doc_id, chunk_index, chunk_text, metadata, embedding, created_at

Usage:
  set OPENAI_API_KEY=sk-...
  python txt_to_chunks_csv.py --input_dir .\input --output chunks.csv

Dependencies:
  pip install openai tqdm
"""

import os
import argparse
import json
import uuid
from pathlib import Path
from datetime import datetime
from time import sleep
from tqdm import tqdm

# new OpenAI v1 client
from openai import OpenAI

# ----------- Chunking (word-based) -----------
def chunk_text_words(text, max_words=350, overlap=50):
    words = text.split()
    n = len(words)
    chunks = []
    start = 0

    while start < n:
        end = min(n, start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap

    return chunks

# ----------- OpenAI v1 embeddings wrapper -----------
# Create client once (reads OPENAI_API_KEY from env)
client = None

def init_openai_client():
    global client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    # OpenAI() will use the provided api_key argument or environment automatically.
    client = OpenAI(api_key=api_key)

def embed_batch(model, texts, max_retries=5, backoff=1.0):
    """
    Uses openai v1 Python client:
      from openai import OpenAI
      client = OpenAI()
      resp = client.embeddings.create(model=..., input=[...])

    Returns: list[list[float]]
    """
    if client is None:
        init_openai_client()

    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            # Each item in resp.data has .embedding attribute
            embeddings = [item.embedding for item in resp.data]
            return embeddings
        except Exception as e:
            print(f"[embed_batch] attempt {attempt+1} failed: {e}")
            if attempt + 1 == max_retries:
                raise
            sleep(backoff * (2 ** attempt))
    raise RuntimeError("Embedding retries exhausted")

# ----------- Main Program -----------
def main(args):
    input_dir = Path(args.input_dir)
    files = list(input_dir.rglob("*.txt"))
    if not files:
        print("No .txt files found in:", input_dir)
        return

    # ensure client is initialized (this will raise if OPENAI_API_KEY missing)
    init_openai_client()

    rows = []
    batch_size = args.batch_size
    model = args.model

    for f in tqdm(files, desc="Processing files"):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue

        doc_id = str(f.relative_to(input_dir))

        chunks = chunk_text_words(
            text,
            max_words=args.max_words,
            overlap=args.overlap
        )

        if not chunks:
            continue

        # Embedding in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            # call embeddings
            embeddings = embed_batch(model, batch)

            for j, chunk_text in enumerate(batch):
                rows.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "chunk_index": i + j,
                    "chunk_text": chunk_text.replace("\n", " ").strip(),
                    "metadata": json.dumps({"source": doc_id}),
                    "embedding": json.dumps(embeddings[j]),
                    "created_at": datetime.utcnow().isoformat() + "Z"
                })

            # polite sleep between batches (adjust if you like)
            if args.sleep > 0:
                sleep(args.sleep)

    # -------- Write CSV ----------
    import csv
    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "id",
            "doc_id",
            "chunk_index",
            "chunk_text",
            "metadata",
            "embedding",
            "created_at"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"✅ Done! Wrote {len(rows)} chunks → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="directory containing .txt files")
    parser.add_argument("--output", default="chunks.csv", help="CSV output path")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--max_words", type=int, default=350, help="max words per chunk (word-based)")
    parser.add_argument("--overlap", type=int, default=50, help="overlap words between chunks (word-based)")
    parser.add_argument("--batch_size", type=int, default=16, help="embedding batch size")
    parser.add_argument("--sleep", type=float, default=0.2, help="sleep seconds between batches")
    args = parser.parse_args()
    main(args)

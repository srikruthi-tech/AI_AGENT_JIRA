#!/usr/bin/env python3
"""
txt_to_chunks_ollama.py

- Read .txt files under --input_dir
- Word-based chunking (no tiktoken)
- Call local Ollama embedding endpoint to get embeddings
- Write CSV with columns: id, doc_id, chunk_index, chunk_text, metadata, embedding, created_at

Usage:
  python txt_to_chunks_ollama.py --input_dir ./input --output chunks.csv --model nomic-embed-text

Notes:
 - Ensure Ollama is installed and the embedding model is pulled and available locally.
 - Default Ollama endpoint: http://localhost:11434
"""

import os
import argparse
import json
import uuid
from pathlib import Path
from datetime import datetime
from time import sleep
from tqdm import tqdm
import csv
import requests

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

# ----------- Ollama embedding call -----------
def call_ollama_embeddings(ollama_url, model, inputs, timeout=60):
    """
    Calls Ollama local embeddings endpoint.
    Expected payload: {"model": model, "input": [...]}
    Expected response: JSON with 'data' list each containing 'embedding' array
    (Adjust if your Ollama version returns slightly different structure.)
    """
    url = f"{ollama_url.rstrip('/')}/v1/embeddings"
    payload = {"model": model, "input": inputs}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    # Try common shapes
    if isinstance(j, dict) and "data" in j:
        embs = []
        for item in j["data"]:
            # support dict or attribute
            if isinstance(item, dict) and "embedding" in item:
                embs.append(item["embedding"])
            elif hasattr(item, "embedding"):
                embs.append(item.embedding)
            else:
                raise RuntimeError("Unexpected embeddings item structure: " + str(item))
        return embs
    # some Ollama setups may return list directly
    if isinstance(j, list):
        return j
    raise RuntimeError("Unexpected Ollama embeddings response: " + json.dumps(j)[:400])

# ----------- Main Program -----------
def main(args):
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print("Input directory does not exist:", input_dir)
        return

    txt_files = list(input_dir.rglob("*.txt"))
    if not txt_files:
        print("No .txt files found in:", input_dir)
        return

    rows = []
    batch_size = args.batch_size
    model = args.model
    ollama_url = args.ollama_url

    detected_dim = None

    for f in tqdm(txt_files, desc="Processing files"):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue

        doc_id = str(f.relative_to(input_dir))
        chunks = chunk_text_words(text, max_words=args.max_words, overlap=args.overlap)

        if not chunks:
            continue

        # send in batches to Ollama
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                embs = call_ollama_embeddings(ollama_url, model, batch, timeout=args.timeout)
            except Exception as e:
                print(f"[error] Ollama embedding call failed: {e}")
                raise

            # set detected_dim on first batch
            if detected_dim is None and len(embs) > 0:
                detected_dim = len(embs[0])
                print(f"[info] Detected embedding dimension = {detected_dim}")

            for j, chunk_text in enumerate(batch):
                emb = embs[j]
                rows.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "chunk_index": i + j,
                    "chunk_text": chunk_text.replace("\n", " ").strip(),
                    "metadata": json.dumps({"source": doc_id, "ingested_by": "ollama-local"}),
                    "embedding": json.dumps(emb),
                    "created_at": datetime.utcnow().isoformat() + "Z"
                })

            if args.sleep > 0:
                sleep(args.sleep)

    # write CSV
    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "doc_id", "chunk_index", "chunk_text", "metadata", "embedding", "created_at"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ Done! Wrote {len(rows)} chunks → {out_path}")
    if detected_dim:
        print(f"Embedding dimension: {detected_dim}")
    else:
        print("Could not detect embedding dimension from response.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="directory containing .txt files")
    parser.add_argument("--output", default="chunks.csv", help="CSV output path")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama embedding model name")
    parser.add_argument("--ollama_url", default="http://localhost:11434", help="Ollama local URL")
    parser.add_argument("--max_words", type=int, default=350, help="max words per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="overlap words between chunks")
    parser.add_argument("--batch_size", type=int, default=8, help="embedding batch size")
    parser.add_argument("--sleep", type=float, default=0.1, help="sleep seconds between batches")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout for Ollama embedding calls")
    args = parser.parse_args()
    main(args)

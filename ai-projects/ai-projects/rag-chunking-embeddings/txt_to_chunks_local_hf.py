#!/usr/bin/env python3
"""
txt_to_chunks_local_hf.py

- Read .txt files under --input_dir
- Word-based chunking (no tiktoken)
- Compute local embeddings via SentenceTransformers (Hugging Face)
- Write CSV with columns: id, doc_id, chunk_index, chunk_text, metadata, embedding, created_at

Usage:
  pip install sentence-transformers tqdm
  python txt_to_chunks_local_hf.py --input_dir ./input --output chunks.csv

Change model with --model argument (default: all-MiniLM-L6-v2)
"""
import os
import argparse
import json
import uuid
from pathlib import Path
from datetime import datetime
from time import sleep
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import numpy as np
import csv

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

# ----------- Embedding helper (SentenceTransformers) -----------
class LocalEmbedder:
    def __init__(self, model_name):
        print(f"[LocalEmbedder] Loading model: {model_name} (this may take a moment)...")
        self.model = SentenceTransformer(model_name)
        # model.get_sentence_embedding_dimension() gives dim
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[LocalEmbedder] Model loaded. Embedding dim = {self.dim}")

    def embed_list(self, texts, batch_size=32):
        """
        Returns a list of lists (float) embeddings, in same order as texts.
        Uses model.encode with convert_to_numpy=True for efficiency.
        """
        # encode returns numpy array (n, dim)
        emb_array = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        # Ensure float lists (python native) for JSON serialization
        return [emb.astype(float).tolist() for emb in emb_array]

# ----------- Main Program -----------
def main(args):
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print("Input directory does not exist:", input_dir)
        return

    # Load embedder
    embedder = LocalEmbedder(args.model)

    txt_files = list(input_dir.rglob("*.txt"))
    if not txt_files:
        print("No .txt files found in:", input_dir)
        return

    rows = []
    batch_size = args.batch_size

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

        # Embed in batches using local embedder
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = embedder.embed_list(batch, batch_size=args.embed_batch_size)

            for j, chunk_text in enumerate(batch):
                rows.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "chunk_index": i + j,
                    "chunk_text": chunk_text.replace("\n", " ").strip(),
                    "metadata": json.dumps({"source": doc_id, "ingested_by": args.ingested_by}),
                    "embedding": json.dumps(embeddings[j]),
                    "created_at": datetime.utcnow().isoformat() + "Z"
                })

            # polite sleep to reduce disk/CPU burst (optional)
            if args.sleep > 0:
                sleep(args.sleep)

    # Write CSV
    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "doc_id", "chunk_index", "chunk_text", "metadata", "embedding", "created_at"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ Done! Wrote {len(rows)} chunks → {out_path}")
    print(f"Embedding dimension used: {embedder.dim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="directory containing .txt files")
    parser.add_argument("--output", default="chunks.csv", help="CSV output path")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformers model name")
    parser.add_argument("--max_words", type=int, default=350, help="max words per chunk (word-based)")
    parser.add_argument("--overlap", type=int, default=50, help="overlap words between chunks (word-based)")
    parser.add_argument("--batch_size", type=int, default=16, help="number of chunks to process per loop for embedding")
    parser.add_argument("--embed_batch_size", type=int, default=32, help="embedder internal batch size")
    parser.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between batches")
    parser.add_argument("--ingested_by", default="local-hf-embed", help="metadata ingested_by value")
    args = parser.parse_args()
    main(args)

# RAG: Chunking and Embeddings

Three scripts that turn `.txt` files into a CSV of text chunks with embeddings, using different embedding providers. The CSV is the knowledge base used by [`prompt-response`](../prompt-response).

| Script | Embeddings from |
|---|---|
| `txt_to_chunks_ollama.py` | Local Ollama (e.g. `nomic-embed-text`) |
| `txt_to_chunks_local_hf.py` | Hugging Face SentenceTransformers (default `all-MiniLM-L6-v2`) |
| `txt_to_chunks_csv.py` | OpenAI embeddings (needs `OPENAI_API_KEY`) |

## How it works
1. Reads every `.txt` file in `--input_dir`.
2. Splits each into overlapping word-based chunks (default 350 words, 50 overlap).
3. Embeds the chunks in batches.
4. Writes a CSV with: `id, doc_id, chunk_index, chunk_text, metadata, embedding, created_at`.

## Usage
```bash
pip install -r requirements.txt

# put your own .txt files in ./input, then:
python txt_to_chunks_ollama.py --input_dir ./input --output chunks.csv --model nomic-embed-text
```
For Ollama, run `ollama pull nomic-embed-text` first. For OpenAI, set `OPENAI_API_KEY` in your environment.

> The `input/` folder is empty in this repo. Add your own text files to try it out.

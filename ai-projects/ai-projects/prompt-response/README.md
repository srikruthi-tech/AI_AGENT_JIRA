# Prompt to Chunks to LLM (Java)

A command-line RAG pipeline in Java. Given a prompt, it finds the most relevant chunks from the CSV made by [`rag-chunking-embeddings`](../rag-chunking-embeddings) and asks a local LLM to answer in structured JSON.

## Pipeline
1. Read the prompt from `input/prompt.txt`.
2. Embed it with Ollama (`/v1/embeddings`).
3. Score every chunk in `chunks.csv` by cosine similarity and keep the top K.
4. Build a prompt that includes those chunks and requires JSON output.
5. Run the model through the Ollama CLI and save the results to `output/`.

## Setup
**Prerequisites:** Java 17+, Maven, Ollama with `ollama pull nomic-embed-text` and `ollama pull llama3.2`.

```bash
# 1. Generate chunks.csv with the embedding script, using the SAME embedding model, and copy it here
# 2. Build
mvn clean package
# 3. Run
java -jar target/prompt-chunk-ollama-1.0.0.jar
```

## Options
Pass as `--key value` arguments:

| Option | Default |
|---|---|
| `prompt_file` | `input/prompt.txt` |
| `chunks_csv` | `chunks.csv` |
| `ollama_url` | `http://localhost:11434` |
| `embed_model` | `nomic-embed-text` |
| `llm_model` | `llama3.2` |
| `top_k` | `5` |
| `max_tokens` | `512` |
| `output_dir` | `output` |

The embedding model must match the one used to build `chunks.csv`, otherwise the dimensions will not line up.

# Jira AI Agent

A chat app where you manage Jira in plain English, for example:

> "Create a ticket for the login bug and assign it to John"

A local LLM (Llama 3.1 via Ollama) decides which Jira action to run, and the backend executes it through the Jira Cloud REST API.

## Features

The agent has tools to:
- create a Jira project
- create a ticket
- assign a ticket to a person (looked up by name)
- add a comment
- view ticket details
- add a watcher
- update ticket status

## How it works

```
React chat UI  --POST /api/chat-->  FastAPI  -->  LangGraph agent  <-->  Ollama (llama3.1)
                                                        |
                                                        v
                                                 Jira Cloud REST API
```

The agent is a LangGraph graph with an `agent` node (the model) and a `tools` node. If the model asks for a tool, the graph runs it and loops back until the model answers. When the model tries to create and assign a ticket in one parallel step, the backend splits the calls so the ticket is created first and the follow-up actions use its new key.

## Setup

**Prerequisites:** Python 3.10+, Node.js 18+, [Ollama](https://ollama.com) with `ollama pull llama3.1`, and a Jira Cloud account with an [API token](https://id.atlassian.com/manage-profile/security/api-tokens).

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your Jira URL, email and API token
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm start
```
Open http://localhost:3000. The UI calls the backend at `http://localhost:8000/api/chat`.

## Notes and limitations
- The system prompt is currently locked to a Jira project with key `SCRUM`; change it in `main.py` for your own project.
- CORS is open (`*`) for local development only. Restrict it before deploying anywhere.
- Credentials are read from `.env` and must never be committed.

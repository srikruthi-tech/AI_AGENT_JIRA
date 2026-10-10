import os
import requests
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the modern tool and graph modules
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# Silence framework warning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN")
AUTH = (JIRA_EMAIL, JIRA_TOKEN)
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

# ----------------------------------------------------
# 🔍 Shared Internal Helper Function
# ----------------------------------------------------
def get_jira_user_by_name(person_name: str):
    search_url = f"{JIRA_URL}/rest/api/3/user/search?query={person_name}"
    response = requests.get(search_url, auth=AUTH, headers=HEADERS)
    if response.status_code == 200 and response.json():
        return response.json()[0]
    return None

# ----------------------------------------------------
# 🛠️ Define Jira Tools for the AI Agent
# ----------------------------------------------------

@tool
def assign_jira_ticket(issue_key: str, assignee: str) -> str:
    """Assign an existing Jira ticket to a person. 
    issue_key is the ticket ID (e.g., 'SCRUM-12'). assignee is the user's name (e.g., 'John Doe')."""
    
    # Update the internal reference from person_name to assignee
    user_data = get_jira_user_by_name(assignee)
    if not user_data:
        return f"Could not find a user matching the name '{assignee}' in Jira."
    
    account_id = user_data["accountId"]
    display_name = user_data["displayName"]

    assign_url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}/assignee"
    payload = {"accountId": account_id}
    
    response = requests.put(assign_url, json=payload, auth=AUTH, headers=HEADERS)
    if response.status_code == 204: 
        return f"Ticket {issue_key} has been successfully assigned to {display_name}."
    return f"Failed to assign ticket: {response.text}"


@tool
def create_new_jira_project(project_key: str, project_name: str, project_lead_name: str = "me") -> str:
    """Create a completely new project workspace board in Jira. 
    project_key must be 3-4 letters uppercase. 
    project_lead_name is the name of the project lead (defaults to your own account if empty)."""
    
    # Step 1: Look up the project lead's Account ID
    lead_id = None
    if project_lead_name and project_lead_name.lower() != "me":
        user_data = get_jira_user_by_name(project_lead_name)
        if user_data:
            lead_id = user_data["accountId"]
            
    # Fallback: If user lookup fails, fetch your own account ID so the tool doesn't crash
    if not lead_id:
        myself_url = f"{JIRA_URL}/rest/api/3/myself"
        myself_resp = requests.get(myself_url, auth=AUTH, headers=HEADERS)
        if myself_resp.status_code == 200:
            lead_id = myself_resp.json()["accountId"]
        else:
            return "Failed project creation: Unable to determine a valid project lead account ID."

    # Step 2: Use the standard Jira Cloud next-gen team-managed Kanban template
    url = f"{JIRA_URL}/rest/api/3/project"
    payload = {
        "key": project_key.upper(),
        "name": project_name,
        "projectTypeKey": "software",
        "projectTemplateKey": "com.atlassian.jira-core-project-templates:jira-core-simplified-kanban-classic",
        "leadAccountId": lead_id
    }
    
    response = requests.post(url, json=payload, auth=AUTH, headers=HEADERS)
    if response.status_code == 201:
        return f"Project '{project_name}' created successfully with Key: {project_key.upper()} and Lead ID: {lead_id}!"
    
    # SAFETY NET: If the classic template still throws an error, try the absolute basic project template
    if response.status_code == 400:
        payload["projectTemplateKey"] = "com.atlassian.jira-core-project-templates:jira-core-simplified-scrum-classic"
        second_attempt = requests.post(url, json=payload, auth=AUTH, headers=HEADERS)
        if second_attempt.status_code == 201:
            return f"Project '{project_name}' created successfully on fallback track with Key: {project_key.upper()}!"

    return f"Jira Cloud API returned an explicit error: {response.text}"


@tool
def create_jira_ticket(project_key: str, summary: str, description: str, issue_type: str = "Task") -> str:
    """Create a new Jira ticket. project_key is usually a 3-4 letter acronym like 'LAB'."""
    
    # 🛡️ SAFETY INTERCEPT: If the LLM generates a blank, lowercase, or missing issue type, 
    # force it to fallback to a valid standard Jira type like 'Task'
    clean_issue_type = issue_type.strip() if issue_type else "Task"
    if clean_issue_type.lower() in ["", "none", "null", "any"]:
        clean_issue_type = "Task"
    else:
        # Capitalize the first letter (e.g., transform 'bug' to 'Bug', 'task' to 'Task')
        clean_issue_type = clean_issue_type.capitalize()

    url = f"{JIRA_URL}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": project_key.upper()},
            "summary": summary,
            "description": {
                "type": "doc", "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": description or "Created via Lab Agent."}]}]
            },
            "issuetype": {"name": clean_issue_type}  # Employs our safe, validated string value
        }
    }
    
    response = requests.post(url, json=payload, auth=AUTH, headers=HEADERS)
    if response.status_code == 201:
        return f"Ticket created successfully! Key: {response.json()['key']}"
        
    return f"Failed to create ticket: {response.text}"


@tool
def add_comment(issue_key: str, comment_text: str) -> str:
    """Add a new comment to an existing Jira ticket using its issue_key (e.g., 'LAB-12')."""
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}/comment"
    payload = {
        "body": {
            "type": "doc", "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment_text}]}]
        }
    }
    response = requests.post(url, json=payload, auth=AUTH, headers=HEADERS)
    if response.status_code == 201:
        return "Comment added successfully."
    return f"Failed to add comment: {response.text}"


@tool
def view_ticket_details(issue_key: str) -> str:
    """Retrieve the status, summary, and comments of an existing Jira ticket."""
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}?fields=status,summary,comment"
    response = requests.get(url, auth=AUTH, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        summary = data["fields"]["summary"]
        status = data["fields"]["status"]["name"]
        
        comments_list = []
        if "comment" in data["fields"] and "comments" in data["fields"]["comment"]:
            for c in data["fields"]["comment"]["comments"]:
                try:
                    text = c["body"]["content"][0]["content"][0]["text"]
                    comments_list.append(text)
                except (KeyError, IndexError):
                    continue
                    
                # Change the multi-line \n string to a flat, comma-separated format
        return f"Ticket Key: {issue_key}. Summary Headline: {summary}. Current Board Status: {status}. Recent Comments: {', '.join(comments_list) if comments_list else 'No comments logged yet'}."

    return f"Failed to fetch ticket: {response.text}"


@tool
def add_ticket_watcher(issue_key: str, person_name: str) -> str:
    """Add a person as a watcher/monitor to a specific Jira ticket."""
    user_data = get_jira_user_by_name(person_name)
    if not user_data:
        return f"Could not find user '{person_name}' to add as a watcher."
        
    account_id = user_data["accountId"]
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}/watchers"
    
    response = requests.post(url, json=account_id, auth=AUTH, headers=HEADERS)
    if response.status_code == 204:
        return f"Successfully added {user_data['displayName']} as a watcher to {issue_key}."
    return f"Failed to add watcher: {response.text}"

@tool
def update_ticket_status(issue_key: str, status_name: str) -> str:
    """Change the status of an existing Jira ticket. 
    issue_key is the ticket ID (e.g., 'LAB-12'). 
    status_name is the target workflow step (e.g., 'In Progress', 'Done', 'To Do')."""
    
    # Step 1: Ask Jira what transition IDs are available for this specific ticket
    transitions_url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}/transitions"
    get_resp = requests.get(transitions_url, auth=AUTH, headers=HEADERS)
    
    if get_resp.status_code != 200:
        return f"Failed to fetch available statuses for {issue_key}: {get_resp.text}"
        
    # Search the available options to find a name match (ignoring upper/lower case)
    transitions_data = get_resp.json().get("transitions", [])
    transition_id = None
    available_names = []
    
    for t in transitions_data:
        available_names.append(t["name"])
        if t["name"].lower() == status_name.lower():
            transition_id = t["id"]
            break
            
    if not transition_id:
        return f"Could not find status '{status_name}'. Available options for this ticket are: {', '.join(available_names)}"

    # Step 2: Trigger the status change using the found transition ID
    payload = {"transition": {"id": transition_id}}
    post_resp = requests.post(transitions_url, json=payload, auth=AUTH, headers=HEADERS)
    
    if post_resp.status_code == 204:
        return f"Ticket {issue_key} successfully moved to status: '{status_name}'."
    return f"Failed to change status: {post_resp.text}"


# ----------------------------------------------------
# 🤖 Build the Agent Engine via State Graph
# ----------------------------------------------------
try:
    model = ChatOllama(model="llama3.1", temperature=0)
except Exception:
    model = ChatOllama(model="llama3:1:latest", temperature=0)

tools = [
    create_new_jira_project, 
    create_jira_ticket, 
    add_comment, 
    view_ticket_details, 
    assign_jira_ticket,  
    add_ticket_watcher,
    update_ticket_status
]

model_with_tools = model.bind_tools(tools)

# 💾 BACKGROUND STORAGE CHECKLIST (Performs Memory Preservation)
# Keeps track of unexecuted tools if the model drops them mid-flight
pending_agent_tasks = {}

def router(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 👁️ Live Console Printing Logic with Advanced Memory Checklist Preservation
def call_model(state: MessagesState):
    print("\n" + "🧠" + "="*60)
    print("🤖 AGENT REASONING STEP")
    print(f"Current conversation turn count: {len(state['messages'])}")
    
    last_msg = state["messages"][-1]
    sender = "User" if last_msg.type == "user" else f"Tool ({last_msg.name})" if last_msg.type == "tool" else "Assistant"
    print(f"  └─ Sent By: {sender}")
    print(f"  └─ Content: {str(last_msg.content)[:150]}...")
    print("="*62)

    # 1. Look for ticket confirmation tokens from a previous creation step
    ticket_key_found = None
    if last_msg.type == "tool" and last_msg.name == "create_jira_ticket" and "Key:" in str(last_msg.content):
        # Extract the new key string from text (e.g., 'SCRUM-6')
        try:
            ticket_key_found = str(last_msg.content).split("Key:")[-1].strip()
            print(f"🔑 Live Ticket Code Captured via State: {ticket_key_found}")
        except Exception:
            pass

    # 2. Invoke the model brain
    response = model_with_tools.invoke(state["messages"])
    
    # 3. IF THE MODEL GENERATES MULTIPLE TOOL CALLS IN ONE GO (Turn 1 Parallel)
    if response.tool_calls and len(response.tool_calls) > 1:
        print("⚠️ PARALLEL BUNDLE DETECTED! Splitting calls to secure sequential workflow layout...")
        
        # Isolate the creation tool call
        creation_call = next((tc for tc in response.tool_calls if tc["name"] == "create_jira_ticket"), None)
        
        if creation_call:
            # Capture and save the remaining tools (assignment, watcher) for later turns
            other_calls = [tc for tc in response.tool_calls if tc["name"] != "create_jira_ticket"]
            pending_agent_tasks["backlog"] = other_calls
            print(f"📝 Cached {len(other_calls)} pending background tasks into graph state memory.")
            
            # Enforce execution boundary: Force response to ONLY contain creation on Turn 1
            response.tool_calls = [creation_call]
            
    # 4. IF THE MODEL EXITS EARLY BUT WE STILL HAVE CACHED BACKLOG TASKS
    elif not response.tool_calls and pending_agent_tasks.get("backlog") and ticket_key_found:
        print("💡 MEMORY PRESERVATION ACTIVATED: Injecting cached tasks with updated ticket key context...")
        backlog = pending_agent_tasks.pop("backlog")
        
        # Re-map the newly generated ticket key to the saved assignment/watcher tool configurations
        for task in backlog:
            if "issue_key" in task["args"]:
                task["args"]["issue_key"] = ticket_key_found
                
        # Force the model's message wrapper to execute these saved tool targets on Turn 3
        response.tool_calls = backlog
        print(f"🚀 Dispatched remaining tasks sequentially: {[t['name'] for t in backlog]}")
    
    print("CNTRL/TARGET AGENT DECISION:")
    if response.tool_calls:
        print(f"  └─ Status: Executing isolated tool step...")
        for tc in response.tool_calls:
            print(f"     👉 Tool: '{tc['name']}' -> Args: {tc['args']}")
    else:
        print(f"  └─ Status: Finalizing text response to UI.")
    print("="*62 + "\n")
    
    return {"messages": [response]}




builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

agent = builder.compile()

# ----------------------------------------------------
# 🌐 FastAPI Web Application Framework
# ----------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat_with_agent(request: ChatRequest):
    try:
        system_instruction = (
            "You are a strict, non-creative Jira Assistant operating in a testing lab.\n"
            "Your target project key is locked to 'SCRUM'. Never generate keys like 'TRN' or 'TRA'.\n\n"
            "🎯 CORE TOOL MAPPING RULES:\n"
            "1. If the user asks to 'view', 'get', 'show', 'check', or 'inspect' details/status of a ticket, "
            "you MUST exclusively call the 'view_ticket_details' tool.\n"
            "2. If a user asks to 'assign', you MUST use the 'assign_jira_ticket' tool.\n"
            "3. If a user asks to 'comment', use 'add_comment'.\n"
            "4. If a user asks to 'change status', use 'update_ticket_status'.\n\n"
            "CRITICAL SEQUENCE RULE:\n"
            "When asked to create and assign a ticket in one go, you MUST call 'create_jira_ticket' first. "
            "Wait for the tool to return a success message containing the new Key code. Never execute in parallel."
        )
        
        inputs = {
            "messages": [
                ("system", system_instruction),
                ("user", request.message)
            ]
        }
        
        config = {"recursion_limit": 15}
        result = agent.invoke(inputs, config=config)
        
        # Extract the last message from the AI response
        final_ai_msg = result["messages"][-1].content
        
        # 🛡️ THE UI FIXED LAYER: Search the graph execution trace history.
        # If the model called 'view_ticket_details', find what the python script returned 
        # and manually append it to the AI's text so the UI is guaranteed to show it!
        raw_tool_data = ""
        for msg in reversed(result["messages"]):
            if msg.type == "tool" and msg.name == "view_ticket_details":
                raw_tool_data = f"\n\n📋 **Live Database Record Details:**\n{msg.content}"
                break
        
        # Combine the conversational answer with the raw database extract
        complete_response = f"{final_ai_msg}{raw_tool_data}"
        
        return {"response": complete_response}
        
    except Exception as e:
        print("\n" + "💥"*25)
        print("🚨 AGENT TERMINAL CRASH LOG:")
        traceback.print_exc()
        print("💥"*25 + "\n")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

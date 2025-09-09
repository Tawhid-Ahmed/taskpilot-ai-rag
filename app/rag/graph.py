from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from app.api.task_tools import create_task,get_tasks,update_task,delete_task,get_task_by_id
from langchain_core.messages import AIMessage,HumanMessage
from app.rag.vector_store import query as rag_query
from langchain.agents import initialize_agent, AgentType
from app.memory.manager import MemoryManager
from langgraph.graph import StateGraph,END
import os
from datetime import datetime, timedelta
from typing import Any

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","llama3.1:latest")

# --- State definition ---
from typing import TypedDict,List

class GraphState(TypedDict):
    user_id: int
    session_id: str
    input: str
    chat_history: List
    output: str


# --- Nodes ---

def input_handler(state: GraphState) -> GraphState:
    """Load memory from DB and prepare state."""
    memory = MemoryManager()
    history = memory.get_coversation(state['user_id'], state['session_id'], limit=20)
    chat_history = [
        HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content)
        for msg in reversed(history)
    ]
    state['chat_history'] = chat_history
    return state


def agent_node(state: GraphState) -> GraphState:
    """Run the LLM with memory context + tools (RAG, CRUD)."""
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    tools = [create_task, get_tasks, update_task, delete_task, get_task_by_id]

    # Retrieve top-k tasks for context
    user_tasks = rag_query(state['user_id'], state['input'], top_k=5)
    context_text = "\n".join([f"{t['title']}: {t.get('description','')}" for t in user_tasks])
    prompt_text = f"You are TaskPilot AI. Here are the user's recent tasks:\n{context_text}\n\nUser message: {state['input']}"

    #initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    # prompt = ChatPromptTemplate.from_messages([
    #      ("system", "You are TaskPilot AI. Use memory + tools to help manage tasks."),
    #      ("system", "If relevant, use CRUD tools (via APIs) to create/update/delete tasks."),
    #      ("user", "{input}")
    # ])

    # chain =prompt | llm
    # response = chain.invoke({
    #     "input": state['input'],
    #      "history": state["chat_history"]
    # })
    # run with chat history context
    response = agent.run(prompt_text)
    # agent.run may return a string or an object with a .content attribute depending on the LLM/toolchain.
    if hasattr(response, "content"):
        out_text = response.content
    else:
        out_text = str(response)
    state['output'] = out_text
    return state


def _parse_date(d: Any):
    """Try to parse a date string into a date object. Returns None on failure."""
    if d is None:
        return None
    if isinstance(d, (datetime,)):
        return d.date()
    if isinstance(d, str):
        # Try common ISO formats
        try:
            return datetime.fromisoformat(d).date()
        except Exception:
            try:
                return datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                return None
    return None


def format_tasks_natural(tasks: list, query: str) -> str:
    """Format a list of task dicts into a human-friendly answer based on the query intent."""
    if not tasks:
        return "I couldn't find any tasks for that query."

    # Basic intent heuristics
    q = query.lower()
    now = datetime.utcnow().date()
    week_end = now + timedelta(days=7)

    # If the query mentions pending/completed, filter by status keys
    if "pending" in q or "incomplete" in q or "not done" in q:
        filtered = [t for t in tasks if str(t.get("status","")).lower() in ("pending","incomplete","todo") or t.get("completed") in (False, "false")]
        if not filtered:
            return "There are no pending tasks."
        lines = [f"{t.get('title')} (due {t.get('dueDate') or t.get('due_date')}) - status: {t.get('status', 'unknown')}" for t in filtered]
        return "Here are the pending tasks:\n" + "\n".join(lines)

    if "due this week" in q or "this week" in q or "due this" in q or "due in the next" in q:
        filtered = []
        for t in tasks:
            d = _parse_date(t.get('dueDate') or t.get('due_date'))
            if d and now <= d <= week_end:
                filtered.append(t)
        if not filtered:
            return "No tasks are due in the next 7 days."
        lines = [f"{t.get('title')} is due on {t.get('dueDate') or t.get('due_date')}" for t in filtered]
        return "Tasks due this week:\n" + "\n".join(lines)

    # Generic summary: list top 5 tasks with due dates
    lines = []
    for t in tasks[:5]:
        due = t.get('dueDate') or t.get('due_date') or 'no due date'
        lines.append(f"{t.get('title')} — due: {due}")
    return "Here are some of your tasks:\n" + "\n".join(lines)


def backend_fetch_node(state: GraphState) -> GraphState:
    """Call the backend get_tasks tool and format a natural language reply.

    This node is used for straightforward retrieval queries (due this week, pending, list tasks).
    """
    # call get_tasks tool directly; it accepts optional params but we just fetch all and filter locally
    try:
        resp = get_tasks({"user_id": state['user_id']})
    except Exception as exc:
        state['output'] = f"Failed to call backend: {exc}"
        return state

    # get_tasks may return an error dict or a list
    if isinstance(resp, dict) and resp.get('error'):
        state['output'] = f"Backend error: {resp.get('error')}"
        return state

    tasks = resp if isinstance(resp, list) else []

    # Handle explicit count requests with optional status filtering
    q = state['input'].lower()
    if any(t in q for t in ("how many", "how many tasks", "count", "number of tasks")):
        # Try to detect a status filter in the user's query (e.g., "new", "pending", "completed")
        status_aliases = {
            "new": "NEW",
            "pending": "PENDING",
            "incomplete": "INCOMPLETE",
            "todo": "TODO",
            "completed": "COMPLETED",
            "done": "COMPLETED",
            "complete": "COMPLETED",
        }

        detected_status = None
        for alias, norm in status_aliases.items():
            if f" {alias} " in f" {q} " or q.endswith(f" {alias}") or q.startswith(f"{alias} "):
                detected_status = norm
                break

        if detected_status:
            filtered = [t for t in tasks if (t.get("status") and str(t.get("status")).upper() == detected_status) or (str(t.get("status") or "").upper() == detected_status)]
            state['output'] = f"You have {len(filtered)} tasks with status {detected_status}."
            return state

        # no status filter detected; return total count
        state['output'] = f"You have {len(tasks)} tasks."
        return state

    # If no tasks returned, fall back to RAG
    if not tasks:
        retrieved = rag_query(state['user_id'], state['input'], top_k=5)
        state['output'] = format_tasks_natural(retrieved, state['input'])
        return state

    # Format response based on query intent
    state['output'] = format_tasks_natural(tasks, state['input'])
    return state


def decision_node(state: GraphState) -> GraphState:
    """Decide whether to use the simple backend-fetch branch or run the agent (RAG + tools).

    Heuristic rules are used for now; can be replaced by a small classifier LLM later.
    """
    q = state['input'].strip()

    def classify_intent(text: str) -> str:
        """Use the LLM as a lightweight 'brain' to classify whether to route to the agent (tools)
        or to the backend/RAG. Returns 'agent' or 'backend'.

        The function tries the LLM first (temperature=0). If the LLM call fails or returns
        an unclear result, a compact heuristic fallback is used.
        """
        # Try LLM classifier first
        try:
            llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
            prompt = (
                "Decide whether the following user message should be handled by calling backend CRUD/tools (return exactly 'agent')\n"
                "or by returning information from memory/vector DB/RAG (return exactly 'backend').\n"
                "Only respond with a single word: agent or backend. If ambiguous, prefer agent.\n\n"
                f"Message: {text}\n\nExamples:\n"
                "Create a task called X -> agent\n"
                "Update task id 36 status pending -> agent\n"
                "Which tasks are due this week? -> backend\n"
                "How many tasks do I have? -> backend\n"
            )
            # ChatOllama supports being called like a callable in this environment; guard with try/except
            resp = llm(prompt)
            ans = str(resp).strip().lower()
            if 'agent' in ans and 'backend' not in ans:
                return 'agent'
            if 'backend' in ans and 'agent' not in ans:
                return 'backend'
        except Exception:
            # LLM-based classification failed; fall through to heuristic
            pass

        # Fallback heuristic (compact and not hard-coded tag list):
        # - If phrase contains clear CRUD verbs or an explicit id reference -> agent
        # - Short natural questions and counting requests -> backend
        text_l = text.lower()
        crud_signals = ["create", "add", "update", "delete", "remove", "complete", "mark", "set"]
        id_signals = ["id:", "id ", "task id", "task_id", "taskid"]
        count_signals = ["how many", "count", "number of"]

        if any(s in text_l for s in crud_signals) or any(s in text_l for s in id_signals):
            return 'agent'
        if any(s in text_l for s in count_signals):
            return 'backend'

        # Default to agent for safety (agents can call backend/tools if needed)
        return 'agent'

    state['branch'] = classify_intent(q)
    return state

def output_handler(state: GraphState) -> GraphState:
    
    """Save new message + AI response into DB memory."""
    memory = MemoryManager()
    memory.save_message(
        state['user_id'],
        state['session_id'],
        "user",
        state['input']
    )
    memory.save_message(
        state['user_id'],
        state['session_id'],
        "ai",
        state['output']
    )
    return state

# --- Build Graph ---
workflow = StateGraph(GraphState)
workflow.add_node("input_handler", input_handler)
workflow.add_node("decision_node", decision_node)
workflow.add_node("agent_node", agent_node)
workflow.add_node("backend_fetch_node", backend_fetch_node)
workflow.add_node("output_handler", output_handler)

workflow.set_entry_point("input_handler")
workflow.add_edge("input_handler", "decision_node")

# branching from decision_node based on state['branch']
workflow.add_conditional_edges(
    "decision_node",
    lambda s: s.get("branch"),
    path_map={"agent": "agent_node", "backend": "backend_fetch_node"},
)

workflow.add_edge("agent_node", "output_handler")
workflow.add_edge("backend_fetch_node", "output_handler")
workflow.add_edge("output_handler", END)
app_graph = workflow.compile()
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.messages import AIMessage,HumanMessage
from app.memory.manager import MemoryManager
from langgraph.graph import StateGraph,END
import os

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

    prompt = ChatPromptTemplate.from_messages([
         ("system", "You are TaskPilot AI. Use memory + tools to help manage tasks."),
         ("system", "If relevant, use CRUD tools (via APIs) to create/update/delete tasks."),
         ("user", "{input}")
    ])

    chain =prompt | llm
    response = chain.invoke({
        "input": state['input'],
         "history": state["chat_history"]
    })
    state['output'] = response.content
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
workflow.add_node("input_handler",input_handler)
workflow.add_node("agent_node",agent_node)
workflow.add_node("output_handler",output_handler)

workflow.set_entry_point("input_handler")
workflow.add_edge("input_handler", "agent_node")
workflow.add_edge("agent_node", "output_handler")
workflow.add_edge("output_handler", END)
app_graph = workflow.compile()
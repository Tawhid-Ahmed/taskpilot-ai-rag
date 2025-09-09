from fastapi import FastAPI, Request
from app.rag.graph import app_graph
from app.schemas import ChatRequest
from fastapi.responses import JSONResponse
from app.api.task_tools import set_auth_token

app = FastAPI(title="TaskPilot AI with RAG and Memory")
@app.post("/chat")
async def chat(rq: ChatRequest, request: Request):
    """Main chat endpoint."""
    # Extract Authorization header from incoming request (if any) and set it for downstream tools
    auth = request.headers.get("authorization")
    set_auth_token(auth)

    state = {
        "user_id": rq.user_id,
        "session_id": rq.session_id,
        "input": rq.message,
        "chat_history": [],
        "output": ""
    }
    try:
        final_state = app_graph.invoke(state)
    finally:
        # Clear token after invocation to avoid leaking between requests
        set_auth_token(None)
    return JSONResponse(content={"response": final_state['output']})
from fastapi import FastAPI , Depends,Request
from app.rag.graph import app_graph
from app.schemas import ChatRequest
from fastapi.responses import JSONResponse

app = FastAPI(title="TaskPilot AI with RAG and Memory")
@app.post("/chat")
async def chat(rq: ChatRequest):
    """Main chat endpoint."""
    state = {
        "user_id": rq.user_id,
        "session_id": rq.session_id,
        "input": rq.message,
        "chat_history": [],
        "output": ""
    }
    final_state = app_graph.invoke(state)
    return JSONResponse(content={"response": final_state['output']})
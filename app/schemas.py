from pydantic import BaseModel
class ChatRequest(BaseModel):
    user_id: int
    session_id: str
    message: str
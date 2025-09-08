from sqlalchemy.orm import Session
from app.memory.models import ConversationMemory
from app.db import SessionLocal

class MemoryManager:
    def __init__(self, db: Session = None):
        self.db = db or SessionLocal()

    def save_message(self, user_id: int, session_id: str, role: str, content: str):
        message = ConversationMemory(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message
    def get_coversation(self,user_id: int, session_id: str,limit: int = 20):
        return (self.db.query(ConversationMemory)
                .filter_by(user_id=user_id, session_id=session_id)
                .order_by(ConversationMemory.created_at.desc())
                .limit(limit)
                .all()
                )
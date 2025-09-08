from sqlalchemy import Column,Integer,String,Text,DateTime,func,ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base

class ConversationMemory(Base):
    __tablename__ = "conversation_memory"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)   # link to Spring Boot User.id
    session_id = Column(String, index=True)     # unique session per chat
    role = Column(String, nullable=False)       # "user" or "ai"
    content = Column(Text, nullable=False)      # message text
    created_at = Column(DateTime, server_default=func.now())
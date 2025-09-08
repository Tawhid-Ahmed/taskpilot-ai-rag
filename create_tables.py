from app.db import Base,engine
from app.memory.models import ConversationMemory

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Tables created successfully.")
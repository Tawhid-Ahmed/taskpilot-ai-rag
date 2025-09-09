from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List,Dict

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Dictionary to hold FAISS indexes for different users (in memory, we will sync with DB)
user_indexes: Dict[int, faiss.IndexFlatL2] = {}
user_task_map: Dict[int, List[Dict]] = {}

def build_index(user_id: int, tasks: List[Dict]):
    """
    tasks: List of dicts with 'id', 'title', 'description'
    """
    """Build or rebuild the FAISS index for a user based on their tasks."""
    if not tasks:
        return None
    texts = [f"{task['title']} {task['description']}" for task in tasks]
    embedings = embedding_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embedings.shape[1])
    index.add(embedings)

    user_indexes[user_id] = index
    user_task_map[user_id] = tasks
    return index

def query(user_id: int, query_text: str, top_k: int = 5) -> List[Dict]:
    """Query the FAISS index for a user and return top_k similar tasks."""
    if user_id not in user_indexes:
        return []

    index = user_indexes[user_id]
    task_list = user_task_map[user_id]
    query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [task_list[i] for i in indices[0] if i < len(task_list)]
    return results
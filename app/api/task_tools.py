import httpx
import os
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("SPRINGBOOT_BASE_URL", "http://localhost:8080/api")
JWT_TOKEN = os.getenv("SPRINGBOOT_JWT_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {JWT_TOKEN}"}

@tool("create_task", return_direct=False)
def create_task(data:dict)->str:
     """Create a new task for the authenticated user. 
    Expects { "title": str, "description": str, "status": str, "dueDate": "YYYY-MM-DD" }"""
     url=f"{BASE_URL}/tasks"
     with httpx.Client() as client:
          response = Client.post(url, json=data, headers=HEADERS)
          if response.status_code == 201:
               return f"Task created successfully: {response.json()}"
          return f"Failed to create task: {response.status_code} - {response.text}"
     

@tool("get_tasks", return_direct=False)
def get_tasks(_:dict =None)->str:
    """Get all tasks for the authenticated user."""
    url = f"{BASE_URL}/tasks"
    with httpx.Client() as client:
        response = client.get(url, headers=HEADERS)
        if response.status_code == 200:
            return f"Tasks retrieved successfully: {response.json()}"
        return f"Failed to retrieve tasks: {response.status_code} - {response.text}"


@tool("update_task", return_direct=False)
def update_task(data:dict)->str:
      """Update a task. Expects { "id": int, "title": str, "description": str, "status": str, "dueDate": "YYYY-MM-DD" }"""
      url=f"{BASE_URL}/tasks/{data['id']}"
      with httpx.Client() as client:
          response = client.put(url, json=data, headers=HEADERS)
          if response.status_code == 200:
              return f"Task updated successfully: {response.json()}"
          return f"Failed to update task: {response.status_code} - {response.text}"
      
@tool("delete_task", return_direct=False)
def delete_task(data:dict)->str:
      """Delete a task. Expects { "id": int }"""
      url=f"{BASE_URL}/tasks/{data['id']}"
      with httpx.Client() as client:
          response = client.delete(url, headers=HEADERS)
          if response.status_code == 204:
              return f"Task deleted successfully."
          return f"Failed to delete task: {response.status_code} - {response.text}"      
      
@tool("get_task_by_id", return_direct=False)
def get_task_by_id(data:dict)->str:
      """Get a task by ID. Expects { "id": int }"""
      url=f"{BASE_URL}/tasks/{data['id']}"
      with httpx.Client() as client:
          response = client.get(url, headers=HEADERS)
          if response.status_code == 200:
              return f"Task retrieved successfully: {response.json()}"
          return f"Failed to retrieve task: {response.status_code} - {response.text}"
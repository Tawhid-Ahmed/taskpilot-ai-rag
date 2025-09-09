import httpx
import os
from langchain_core.tools import tool
from dotenv import load_dotenv
from app.rag.vector_store import build_index

load_dotenv()

BASE_URL = os.getenv("SPRINGBOOT_BASE_URL", "http://localhost:8080/api")
JWT_TOKEN = os.getenv("SPRINGBOOT_JWT_TOKEN", "")
# Only set Authorization header when a token is provided to avoid sending an empty "Bearer " value
HEADERS = {}
if JWT_TOKEN:
    HEADERS["Authorization"] = f"Bearer {JWT_TOKEN}"


def set_auth_token(token: str | None):
    """Runtime helper to set Authorization header (used when token is passed in request headers)."""
    global HEADERS
    if token:
        # Accept either raw token or 'Bearer <token>' and normalize
        if token.lower().startswith("bearer "):
            token = token.split(" ", 1)[1]
        HEADERS["Authorization"] = f"Bearer {token}"
    else:
        HEADERS.pop("Authorization", None)


def refresh_user_index(user_id: int):
    """Fetch tasks from Spring Boot and rebuild FAISS index"""
    url = f"{BASE_URL}/tasks"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=HEADERS)
            if resp.status_code == 200:
                tasks = resp.json()
                build_index(user_id, tasks)
    except httpx.RequestError as exc:
        # Fail silently for background index refresh; log would be better in production
        return f"Failed to refresh index: {exc}"


import json

@tool("create_task", return_direct=True)
def create_task(data: dict | str | None = None):
    """Create a task. Accepts dict or JSON string. Defensive parsing to tolerate agent inputs."""
    if data is None:
        return {"error": "No data provided to create_task."}

    # Only parse JSON if it's a string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            # If agent passed a descriptive string, treat as empty input
            data = {}

    # Optional: convert snake_case to camelCase
    if isinstance(data, dict) and "due_date" in data:
        data["dueDate"] = data.pop("due_date")

    url = f"{BASE_URL}/tasks"
    try:
        with httpx.Client() as client:
            # Idempotency check: fetch existing tasks and try to find a match
            try:
                existing_resp = client.get(url, headers=HEADERS)
                if existing_resp.status_code == 200:
                    existing_tasks = existing_resp.json()
                    # normalize keys for comparison
                    def _norm(t):
                        return {
                            "title": (t.get("title") or "").strip().lower(),
                            "due": (t.get("dueDate") or t.get("due_date") or "").strip(),
                            "desc": (t.get("description") or "").strip().lower(),
                        }

                    target = {"title": (data.get("title") or "").strip().lower(),
                              "due": (data.get("dueDate") or data.get("due_date") or "").strip(),
                              "desc": (data.get("description") or "").strip().lower()}

                    for t in existing_tasks:
                        n = _norm(t)
                        # match title + due date as primary key; description is optional
                        if n["title"] == target["title"] and (not target["due"] or n["due"] == target["due"]):
                            # if description provided, also check it; otherwise accept
                            if target["desc"] and n["desc"] != target["desc"]:
                                continue
                            # Found an existing similar task — return a confirmation string
                            due = t.get("dueDate") or t.get("due_date") or "no due date"
                            return f"Task already exists: '{t.get('title')}' (id: {t.get('id')}) due {due}."
            except httpx.RequestError:
                # if fetch fails, continue to attempt creation (best-effort)
                pass

            response = client.post(url, json=data, headers=HEADERS)
            if response.status_code == 201:
                created = response.json()
                due = created.get("dueDate") or created.get("due_date") or "no due date"
                return f"Created task '{created.get('title')}' (id: {created.get('id')}) due {due}."
            return {"error": f"Failed to create task: {response.status_code}", "details": response.text}
    except httpx.RequestError as exc:
        return {"error": "Failed to create task (network error)", "details": str(exc)}

@tool("get_tasks")
def get_tasks(data: dict | str | None = None):
    """Fetch all tasks for the authenticated user. Accepts optional params as dict or JSON string."""
    # Agents sometimes pass natural language strings; try to parse JSON when given a string.
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            data = {}

    data = data or {}  # default to empty dict
    url = f"{BASE_URL}/tasks"
    try:
        with httpx.Client() as client:
            response = client.get(url, params=data, headers=HEADERS)
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to fetch tasks: {response.status_code}", "details": response.text}
    except httpx.RequestError as exc:
        return f"Failed to fetch tasks (network error): {exc}"

@tool("update_task", return_direct=True)
def update_task(data: dict | str | None = None):
    """Update a task. Accepts dict or JSON string."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            # try to interpret simple id string
            try:
                data = {"id": int(data)}
            except Exception:
                return "Invalid JSON provided to update_task."

    # Accept alternative id keys used by agents
    if isinstance(data, dict):
        if "id" not in data:
            if "task_id" in data:
                try:
                    data["id"] = int(data.pop("task_id"))
                except Exception:
                    return "Invalid 'task_id' provided to update_task."
            elif "taskId" in data:
                try:
                    data["id"] = int(data.pop("taskId"))
                except Exception:
                    return "Invalid 'taskId' provided to update_task."

    if not data or "id" not in data:
        return "Missing 'id' in update_task input."

    url = f"{BASE_URL}/tasks/{data['id']}"
    try:
        with httpx.Client(timeout=10.0) as client:
            # Fetch existing task to merge fields (backend PUT expects full object)
            try:
                get_resp = client.get(url, headers=HEADERS)
            except httpx.RequestError as exc:
                return f"Failed to retrieve existing task before update: {exc}"

            if get_resp.status_code == 200:
                existing = get_resp.json()
                # Merge: preserve existing values and overwrite with any provided fields
                merged = existing.copy()
                # Accept either snake_case or camelCase keys from agent
                for k, v in (data or {}).items():
                    if k == 'due_date':
                        merged['dueDate'] = v
                    elif k == 'task_id' or k == 'taskId':
                        # ignore
                        continue
                    else:
                        merged[k] = v

                # Normalize status to uppercase if present
                if 'status' in merged and isinstance(merged['status'], str):
                    merged['status'] = merged['status'].upper()

                response = client.put(url, json=merged, headers=HEADERS)
            else:
                # couldn't fetch existing, fall back to original update attempt
                response = client.put(url, json=data, headers=HEADERS)

            if response.status_code == 200:
                updated = response.json()
                refresh_user_index(data.get("user_id", 1))
                return f"Updated task '{updated.get('title')}' (id: {updated.get('id')})."
            return {"error": f"Failed to update task: {response.status_code}", "details": response.text}
    except httpx.RequestError as exc:
        return f"Failed to update task (network error): {exc}"


@tool("delete_task", return_direct=True)
def delete_task(data: dict | str | None = None):
    """Delete a task. Accepts dict or JSON string."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return "Invalid JSON provided to delete_task."

    if not data or "id" not in data:
        return "Missing 'id' in delete_task input."

    url = f"{BASE_URL}/tasks/{data['id']}"
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.delete(url, headers=HEADERS)
            if response.status_code == 204:
                refresh_user_index(data.get("user_id", 1))
                return f"Deleted task id {data.get('id')}"
            return {"error": f"Failed to delete task: {response.status_code}", "details": response.text}
    except httpx.RequestError as exc:
        return f"Failed to delete task (network error): {exc}"


@tool("get_task_by_id", return_direct=False)
def get_task_by_id(data: dict | str | None = None):
    """Get a task by ID. Accepts dict or JSON string."""
    # Normalize inputs: accept JSON string, numeric string, or dict with id/task_id/taskId
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            # try to extract integer id from string
            try:
                data = {"id": int(data)}
            except Exception:
                # could be a bare token or malformed input
                return "Invalid input for get_task_by_id."

    if isinstance(data, dict):
        if "id" not in data:
            if "task_id" in data:
                try:
                    data["id"] = int(data.pop("task_id"))
                except Exception:
                    return "Invalid 'task_id' provided to get_task_by_id."
            elif "taskId" in data:
                try:
                    data["id"] = int(data.pop("taskId"))
                except Exception:
                    return "Invalid 'taskId' provided to get_task_by_id."

    if not data or "id" not in data:
        return "Missing 'id' in get_task_by_id input."

    url = f"{BASE_URL}/tasks/{data['id']}"
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=HEADERS)
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to retrieve task: {response.status_code}", "details": response.text}
    except httpx.RequestError as exc:
        return f"Failed to retrieve task (network error): {exc}"

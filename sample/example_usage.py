"""
End-to-end usage example for the Smart API Agent.

Run this after starting the server:
  cd smart-api-agent && python server.py

Then in a separate terminal:
  python example_usage.py
"""
from __future__ import annotations

import asyncio
import json
import httpx

BASE_URL = "http://localhost:8000"
TENANT_ID = "acme-corp"
SESSION_ID = "user-session-001"


# ── Sample tenant registration ───────────────────────────────────────────────

REGISTER_PAYLOAD = {
    "tenant_id": TENANT_ID,
    "company_name": "Acme Corporation",
    "company_info": (
       
        "It provides resources like posts, comments, albums, photos, todos, and users. "
        "No authentication required. Base URL: https://jsonplaceholder.typicode.com/"
    ),
    "api_specs": [
        {
            "name": "list_posts",
            "description": "Retrieve all posts or filter by userId.",
            "method": "GET",
            "url": "https://jsonplaceholder.typicode.com/posts",
            "query_params": {
                "userId": "int"
            },
        },
        {
            "name": "get_post",
            "description": "Retrieve a single post by ID.",
            "method": "GET",
            "url": "https://jsonplaceholder.typicode.com/posts/{id}",
            "path_params": ["id"],
        },
        {
            "name": "create_post",
            "description": "Create a new post.",
            "method": "POST",
            "url": "https://jsonplaceholder.typicode.com/posts",
            "request_body": {
                "title": "string",
                "body": "string",
                "userId": "int",
            },
        },
        {
            "name": "update_post",
            "description": "Update an existing post بالكامل (PUT).",
            "method": "PUT",
            "url": "https://jsonplaceholder.typicode.com/posts/{id}",
            "path_params": ["id"],
            "request_body": {
                "id": "int",
                "title": "string",
                "body": "string",
                "userId": "int",
            },
        },
        {
            "name": "patch_post",
            "description": "Partially update a post.",
            "method": "PATCH",
            "url": "https://jsonplaceholder.typicode.com/posts/{id}",
            "path_params": ["id"],
            "request_body": {
                "title": "string",
                "body": "string",
                "userId": "int",
            },
        },
        {
            "name": "delete_post",
            "description": "Delete a post by ID.",
            "method": "DELETE",
            "url": "https://jsonplaceholder.typicode.com/posts/{id}",
            "path_params": ["id"],
        },
        {
            "name": "list_users",
            "description": "Retrieve all users.",
            "method": "GET",
            "url": "https://jsonplaceholder.typicode.com/users",
        },
        {
            "name": "list_comments",
            "description": "Retrieve comments, optionally filtered by postId.",
            "method": "GET",
            "url": "https://jsonplaceholder.typicode.com/comments",
            "query_params": {
                "postId": "int"
            },
        },
        {
            "name": "list_albums",
            "description": "Retrieve albums, optionally filtered by userId.",
            "method": "GET",
            "url": "https://jsonplaceholder.typicode.com/albums",
            "query_params": {
                "userId": "int"
            },
        },
        {
            "name": "list_photos",
            "description": "Retrieve photos, optionally filtered by albumId.",
            "method": "GET",
            "url": "https://jsonplaceholder.typicode.com/photos",
            "query_params": {
                "albumId": "int"
            },
        },
        {
            "name": "list_todos",
            "description": "Retrieve todos, optionally filtered by userId.",
            "method": "GET",
            "url": "https://jsonplaceholder.typicode.com/todos",
            "query_params": {
                "userId": "int"
            },
        },
    ],
}


async def register_tenant():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE_URL}/register", json=REGISTER_PAYLOAD)
        r.raise_for_status()
        print("✅ Tenant registered:", r.json())


async def send_message(message: str):
    """Stream a chat message and handle clarification pauses."""
    print(f"\n👤 User: {message}")
    payload = {"tenant_id": TENANT_ID, "message": message}

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", f"{BASE_URL}/chat/{SESSION_ID}", json=payload) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    _handle_event(event)


async def resume_with_reply(reply: str):
    """Resume after a human-in-the-loop pause."""
    print(f"\n👤 User (reply): {reply}")
    payload = {"tenant_id": TENANT_ID, "message": reply}

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", f"{BASE_URL}/resume/{SESSION_ID}", json=payload) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    _handle_event(event)


def _handle_event(event: dict):
    t = event.get("type")
    if t == "node_start":
        print(f"\n⚙️  [{event['node']}]", end=" ", flush=True)
    elif t == "token":
        print(event["content"], end="", flush=True)
    elif t == "clarification_needed":
        print(f"\n🤖 Assistant (clarification): {event['content']}")
    elif t == "done":
        print("\n✅ Done.\n")


async def main():
    await register_tenant()

    # 1. RAG query (still valid – based on company_info)
    await send_message("What is this API used for?")

    # 2. Simple GET
    await send_message("Show me all posts created by user 1.")

    # 3. Complex analytical query → coder agent
    await send_message(
        "How many posts were created by each user?"
    )

    # 4. CREATE with missing info → human in the loop
    await send_message("Create a new post about a network outage.")
    # Graph pauses, user provides missing info
    await resume_with_reply("Title is 'Network Issue', body is 'System outage in region', userId is 1.")

    # 5. UPDATE (instead of real-world DELETE use-case)
    await send_message("Update post number 42 with title 'Updated Title' and body 'Updated content'.")

    # 6. DELETE
    await send_message("Delete post number 42.")


if __name__ == "__main__":
    asyncio.run(main())

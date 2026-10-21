"""ADK agent that answers from a Supabase table.

This is a separate app from the policy agent. Run it on port 8502:

    streamlit run records_app.py --server.port 8502

The tool reads SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE, and
SUPABASE_SEARCH_COLUMNS from the environment. It does not embed those secrets.
The live table is public.employees (id, name, department, location).
"""

import asyncio
import concurrent.futures
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import Agent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors
from google.genai import types

from policy_agent.agent import _event_steps

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", "False")

MODEL = "gemini-3.8-flash"
APP_NAME = "records_agent"
MAX_LLM_CALLS = 6


def _identifier(name: str) -> str:
    """Keep a table or column name to letters, digits, and underscores."""
    cleaned = "".join(char for char in name if char.isalnum() or char == "_")
    return cleaned or "employees"


def lookup_records(search: str) -> dict:
    """Look up employees whose name, department, or location contains the search words.

    Call this before answering any question about employees.
    Do not invent rows.

    Args:
        search: Words to match, taken from the user's question.
            Examples: a name, a department, or a city.

    Returns:
        status "ok" and the matching rows, or status "error" and a message.
        Each row has id, name, department, and location.
    """
    base_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_KEY", "").strip()
    table = _identifier(os.getenv("SUPABASE_TABLE", "employees"))
    raw_columns = os.getenv("SUPABASE_SEARCH_COLUMNS", "name,department,location")
    columns = [
        _identifier(part) for part in raw_columns.split(",") if part.strip()
    ] or ["name", "department", "location"]
    if not base_url or not key:
        return {
            "status": "error",
            "rows": [],
            "message": "SUPABASE_URL or SUPABASE_KEY is not set.",
        }
    if not search.strip():
        return {
            "status": "error",
            "rows": [],
            "message": "Search text is empty.",
        }
    safe_search = (
        search.replace("%", "")
        .replace("*", "")
        .replace(",", " ")
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
        .strip()
    )
    match = ",".join(f"{column}.ilike.*{safe_search}*" for column in columns)
    query = urllib.parse.urlencode(
        {
            "select": "*",
            "or": f"({match})",
            "limit": "10",
        }
    )
    request = urllib.request.Request(
        f"{base_url}/rest/v1/{table}?{query}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            rows = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return {
            "status": "error",
            "rows": [],
            "message": f"Supabase returned HTTP {exc.code}: {body}",
        }
    except Exception as exc:
        return {"status": "error", "rows": [], "message": str(exc)}
    if not isinstance(rows, list):
        return {
            "status": "error",
            "rows": [],
            "message": "Supabase did not return a list of rows.",
        }
    return {"status": "ok", "table": table, "rows": rows}


root_agent = Agent(
    name="records_agent",
    model=MODEL,
    description="Looks up employees stored in the Supabase employees table.",
    instruction=(
        "Goal: answer one question using employee rows stored in Supabase.\n"
        "Each row has id, name, department, location, remote_eligible, "
        "and vacation_days_left.\n"
        "Constraints:\n"
        "- Call lookup_records once before you answer.\n"
        "- Pass a name, department, or city from the question as search.\n"
        "- Use only the rows that tool returns. Do not use outside knowledge.\n"
        "- If status is error, say the lookup failed and include the message.\n"
        "- If rows is empty, say no matching employee was found.\n"
        "Done: one short answer that names each matching employee and their "
        "department and location, then stop."
    ),
    tools=[lookup_records],
)

# A2A server. The policy router calls this card; it does not import this agent.
app = to_a2a(root_agent, host="127.0.0.1", port=8002)


async def run_question(message: str) -> dict:
    """Run one question against the records agent and collect Think / Act / Observe."""
    service = InMemorySessionService()
    session = await service.create_session(app_name=APP_NAME, user_id="user1")
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=service)
    content = types.Content(role="user", parts=[types.Part(text=message)])
    steps: list[dict] = []
    answer = "(no response)"
    error = ""
    try:
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
            run_config=RunConfig(max_llm_calls=MAX_LLM_CALLS),
        ):
            new_steps, logged = _event_steps(event)
            for step in new_steps:
                print(f"{step['label']} ({step['author']}): {step['detail']}")
            steps.extend(new_steps)
            if logged:
                answer = logged
    except LlmCallsLimitExceededError as exc:
        error = str(exc)
        steps.append({"label": "Observe", "author": "records_agent", "detail": error})
    except errors.APIError as exc:
        error = str(exc)
        steps.append({"label": "Observe", "author": "records_agent", "detail": error})
    return {"answer": answer, "steps": steps, "error": error}


def run_question_sync(message: str, timeout: int = 180) -> dict:
    """Run one question from Streamlit, which already has its own event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run_question(message)).result(timeout=timeout)

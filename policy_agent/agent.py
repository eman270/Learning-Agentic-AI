"""One ADK agent for the Northwind policy question.

Job: answer one question from company-document excerpts, cite the chunk id,
and say the documents do not contain the answer when search returns nothing.

The router sends document questions to ``document_agent`` (Pinecone via
``retrieved_excerpts``) and database questions to the records agent over A2A.
The card URL is ``RECORDS_AGENT_CARD_URL``. The deployed records service
card is https://records-agent.onrender.com/.well-known/agent-card.json.

Patterns copied from:
- ``adk-multi-agent-systems/demo1_routing.py`` (one Agent, a plain function
  tool, InMemorySessionService, Runner, run_async)
- ``ADK-Practice/multi_tool_agent/agent.py`` (``root_agent`` for ``adk web``,
  load the capstone ``.env``)

Run from ``Learning-Agentic-AI``:

    python -m policy_agent.agent
    streamlit run ui/agent.py
    adk web --max_llm_calls 6
"""

import asyncio
import concurrent.futures
import os
import sys

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", "False")

# Demo 1 uses gemini-2.5-flash. This key is a new user, and that model
# returns 404 with a pointer to gemini-3.8-flash.
MODEL = "gemini-3.8-flash"
APP_NAME = "policy_agent"
# One model call to choose the tool, one to write the answer, plus a little
# room if the model retries. ADK's default is 500, which is too open.
MAX_LLM_CALLS = 6


def search_company_docs(question: str) -> dict:
    """Search the company document index for excerpts that answer a question.

    Call this for any question about company policy, the handbook, expenses,
    security, or product specs. Do not answer those from memory.

    Args:
        question: The user's question, in their own words.

    Returns:
        status "ok" and excerpts, each with text, document_id, and chunk_id.
        status "error" and a message when search fails. An empty excerpts
        list means the index had no match.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from main import retrieved_excerpts

        documents = retrieved_excerpts(question)
    except Exception as exc:
        detail = getattr(exc, "detail", None) or str(exc)
        return {
            "status": "error",
            "question": question,
            "excerpts": [],
            "message": str(detail),
        }
    excerpts = [
        {
            "text": document.page_content,
            "document_id": str(document.metadata.get("document_id", "")),
            "chunk_id": str(document.metadata.get("chunk_id", "")),
        }
        for document in documents
    ]
    return {"status": "ok", "question": question, "excerpts": excerpts}


document_agent = Agent(
    name="document_agent",
    model=MODEL,
    description="Searches company policy documents such as the handbook, expenses, security, and product specs.",
    instruction=(
        "Goal: answer one question about Northwind company documents.\n"
        "Constraints:\n"
        "- Call search_company_docs once before you answer.\n"
        "- Use only what that tool returns. Do not use outside knowledge.\n"
        "- If status is error, say the search failed and include the message.\n"
        "- If excerpts is empty, tell the user you do not have enough "
        "information in the provided documents.\n"
        "- When excerpts are present, answer from them and cite each chunk_id "
        "you used, such as [POL-101-3].\n"
        "Done: one short answer to the user, then stop."
    ),
    tools=[search_company_docs],
)

# The records agent runs in its own process. This client only knows the card URL.
records_agent = RemoteA2aAgent(
    name="records_agent",
    agent_card=os.getenv(
        "RECORDS_AGENT_CARD_URL",
        "https://records-agent.onrender.com/.well-known/agent-card.json",
    ),
    description="Looks up employees in the Supabase employees table by name, department, or location.",
)

root_agent = Agent(
    name="policy_router",
    model=MODEL,
    description="Chooses whether a question goes to company documents or the records database.",
    instruction=(
        "You route one question. Never answer it yourself.\n"
        "- Company policy, the handbook, expenses, security, facilities, or "
        "product specs: transfer to document_agent.\n"
        "- Employees, a person's name, a department, a location, or what the "
        "database says: transfer to records_agent.\n"
        "Transfer once, then give the user that specialist's answer."
    ),
    sub_agents=[document_agent, records_agent],
)


def _event_steps(event) -> tuple[list[dict], str | None]:
    """Turn one ADK event into Think, Act, Observe, or Answer steps."""
    author = getattr(event, "author", "agent")
    answer = None
    steps: list[dict] = []
    content = getattr(event, "content", None)
    parts = list(content.parts) if content and content.parts else []

    if not parts and getattr(event, "error_message", None):
        steps.append(
            {"label": "Observe", "author": author, "detail": str(event.error_message)}
        )
        return steps, None

    for part in parts:
        function_call = getattr(part, "function_call", None)
        function_response = getattr(part, "function_response", None)
        text = getattr(part, "text", None)
        if text and getattr(part, "thought", False):
            steps.append({"label": "Think", "author": author, "detail": text})
        elif function_call:
            args = dict(function_call.args) if function_call.args else {}
            steps.append(
                {
                    "label": "Act",
                    "author": author,
                    "detail": f"{function_call.name}({args})",
                }
            )
        elif function_response:
            steps.append(
                {
                    "label": "Observe",
                    "author": author,
                    "detail": (
                        f"{function_response.name} -> {function_response.response}"
                    ),
                }
            )
        elif text and event.is_final_response():
            steps.append({"label": "Answer", "author": author, "detail": text})
            answer = text
        elif text:
            steps.append({"label": "Think", "author": author, "detail": text})
    return steps, answer


async def run_question(message: str) -> dict:
    """Run one question. Returns the answer, the step list, and any error.

    Session: the memory for one conversation. It stores the user message,
    the tool call, the tool result, and the answer under app name + user id
    + session id. InMemorySessionService keeps that only in RAM.

    Runner: the loop that connects the agent and the session. It loads this
    session, gives the new message to policy_qa_agent, lets the agent call
    search_company_docs, saves each step, and yields those steps here.
    """
    service = InMemorySessionService()
    session = await service.create_session(app_name=APP_NAME, user_id="user1")
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=service)
    content = types.Content(role="user", parts=[types.Part(text=message)])
    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS)
    steps: list[dict] = []
    answer = "(no response)"
    error = ""
    try:
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
            run_config=run_config,
        ):
            new_steps, logged = _event_steps(event)
            for step in new_steps:
                print(f"{step['label']} ({step['author']}): {step['detail']}")
            steps.extend(new_steps)
            if logged:
                answer = logged
    except LlmCallsLimitExceededError as exc:
        error = str(exc)
        print(f"Stopped: {error}")
        steps.append({"label": "Observe", "author": "policy_qa_agent", "detail": error})
    except errors.APIError as exc:
        error = str(exc)
        print(f"Stopped: {error}")
        steps.append({"label": "Observe", "author": "policy_qa_agent", "detail": error})
    return {"answer": answer, "steps": steps, "error": error}


def run_question_sync(message: str, timeout: int = 180) -> dict:
    """Run one question from Streamlit, which already has its own event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run_question(message)).result(timeout=timeout)


async def ask(message: str) -> str:
    """Run one question, print the steps, and return the final answer."""
    print(f"User: {message}")
    print(f"Step limit: {MAX_LLM_CALLS} model calls")
    result = await run_question(message)
    return result["answer"]


async def main() -> None:
    await ask("How many remote days are allowed?")


if __name__ == "__main__":
    asyncio.run(main())

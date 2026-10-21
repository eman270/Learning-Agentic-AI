"""Minimal Streamlit page for the Northwind policy agent.

Run from Learning-Agentic-AI:

    streamlit run ui/agent.py
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

from policy_agent.agent import run_question_sync

st.set_page_config(page_title="Policy agent", layout="centered")
st.title("Policy agent")
st.write(
    "Ask one question. The router sends policy questions to the document "
    "search and employee questions to the records agent on port 8002."
)

question = st.text_area(
    "Question",
    height=120,
    placeholder="How many remote days are allowed?",
)

if st.button("Run", type="primary"):
    if not question.strip():
        st.warning("Type a question first.")
        st.stop()
    missing = [
        name
        for name in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        st.error("Missing environment variables: " + ", ".join(missing))
    else:
        with st.spinner("Running the agent..."):
            try:
                result = run_question_sync(question.strip())
            except Exception as exc:
                result = {
                    "answer": "(no response)",
                    "steps": [
                        {
                            "label": "Observe",
                            "author": "policy_qa_agent",
                            "detail": str(exc),
                        }
                    ],
                    "error": str(exc),
                }
        st.session_state["agent_result"] = result

result = st.session_state.get("agent_result")
if result:
    st.subheader("Steps")
    for step in result["steps"]:
        label = step["label"]
        detail = step["detail"]
        if label == "Think":
            st.info(f"**Think.** {detail}")
        elif label == "Act":
            st.warning(f"**Act.** {detail}")
        elif label == "Observe":
            with st.expander("Observe", expanded=True):
                st.code(detail, language="text")
        elif label == "Answer":
            continue
        else:
            st.write(f"**{label}.** {detail}")

    st.subheader("Final answer")
    if result.get("error") and result["answer"] == "(no response)":
        st.error(result["error"])
    else:
        st.success(result["answer"])

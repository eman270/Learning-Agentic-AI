import os

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
)
from pydantic import ValidationError

from main import iter_answer_events

load_dotenv()

st.set_page_config(page_title="Ask", layout="centered")
st.title("Ask")
st.write("Ask a question. The answer appears as the model writes it.")

if "question" not in st.session_state:
    st.session_state.question = ""
if "submitted_question" not in st.session_state:
    st.session_state.submitted_question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "error" not in st.session_state:
    st.session_state.error = ""


def queue_question() -> None:
    """Save the typed question and clear the box before Streamlit reruns.

    The next run reads submitted_question, streams the answer, then clears
    that field so the same question is not sent again.
    """
    st.session_state.submitted_question = st.session_state.question.strip()
    st.session_state.question = ""
    st.session_state.answer = ""
    st.session_state.error = ""


st.text_area(
    "Question",
    height=160,
    placeholder="Ask a question...",
    key="question",
)

st.button(
    "Ask",
    type="primary",
    disabled=not st.session_state.question.strip(),
    on_click=queue_question,
)

submitted = st.session_state.submitted_question
if submitted:
    st.session_state.submitted_question = ""
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state.error = "OPENAI_API_KEY is not set. Add it to a .env file in this folder."
    elif not os.getenv("PINECONE_API_KEY"):
        st.session_state.error = "PINECONE_API_KEY is not set. Add it to a .env file in this folder."
    else:
        slot = st.empty()
        try:
            with st.spinner("Getting an answer..."):
                for kind, payload in iter_answer_events(submitted):
                    if kind == "summary":
                        rendered = payload.model_dump_json(indent=2)
                        st.session_state.answer = rendered
                        slot.code(rendered, language="json")
                    elif payload.answer:
                        slot.markdown(payload.answer)
        except APIError:
            st.session_state.error = "Failed to get an answer."
        except (
            ValidationError,
            LengthFinishReasonError,
            ContentFilterFinishReasonError,
            ValueError,
        ):
            st.session_state.error = "The structured output could not be validated."

if st.session_state.error:
    st.error(st.session_state.error)
elif st.session_state.answer and not submitted:
    st.code(st.session_state.answer, language="json")

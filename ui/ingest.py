import os

import streamlit as st
from dotenv import load_dotenv
from fastapi import HTTPException
from openai import APIError

from main import store_document

load_dotenv()

st.set_page_config(page_title="Ingest", layout="centered")
st.title("Ingest")
st.write("Save plain text in Pinecone. The document id becomes the chunk id prefix.")

if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = ""
if "ingest_error" not in st.session_state:
    st.session_state.ingest_error = ""

with st.form("ingest"):
    document_id = st.text_input("Document id", placeholder="POL-101")
    text = st.text_area("Text", height=240, placeholder="Paste the document text...")
    submitted = st.form_submit_button("Ingest", type="primary")

if submitted:
    st.session_state.ingest_result = ""
    st.session_state.ingest_error = ""
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state.ingest_error = "OPENAI_API_KEY is not set. Add it to a .env file in this folder."
    elif not os.getenv("PINECONE_API_KEY"):
        st.session_state.ingest_error = "PINECONE_API_KEY is not set. Add it to a .env file in this folder."
    else:
        try:
            with st.spinner("Saving chunks..."):
                result = store_document(document_id, text)
            st.session_state.ingest_result = result.model_dump_json(indent=2)
        except HTTPException as exc:
            st.session_state.ingest_error = str(exc.detail)
        except APIError:
            st.session_state.ingest_error = "Failed to create embeddings."

if st.session_state.ingest_error:
    st.error(st.session_state.ingest_error)
elif st.session_state.ingest_result:
    st.code(st.session_state.ingest_result, language="json")

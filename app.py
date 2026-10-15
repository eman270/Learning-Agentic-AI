import streamlit as st
from dotenv import load_dotenv

load_dotenv()

home = st.Page("ui/home.py", title="Home", default=True, visibility="hidden")
ask = st.Page("ui/ask.py", title="Ask", url_path="ask")
ingest = st.Page("ui/ingest.py", title="Ingest", url_path="ingest")

st.navigation([home, ask, ingest]).run()

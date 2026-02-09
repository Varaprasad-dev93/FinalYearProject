from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st
def get_llm():
    api_key = st.secrets['gemini']
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.1,
        # max_output_tokens=1400,
        google_api_key="AIzaSyBKubRLNwJBQXMyNPGV10QPrKNjS-QJ8dg"
    )

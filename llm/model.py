from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st
def get_llm():
    api_key=os.environ["GOOGLE_API_KEY"]
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.1,
        max_output_tokens=16384,
        google_api_key=api_key,
        thinking_budget=0
    )

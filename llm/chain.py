# llm/chain.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from .model import get_llm

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # Keep only last 3 exchanges (k=3 like the paper)
    history = store[session_id]
    if len(history.messages) > 6:
        history.messages = history.messages[-6:]
    return history

def inject_initial_prompt(context_id: str, system_prompt: str):
    """
    Injects dataset schema as a human/AI exchange so Gemini
    sees it as valid conversation history — not a SystemMessage.
    """
    history = get_session_history(context_id)
    if len(history.messages) == 0:  # only inject once
        history.add_message(HumanMessage(content=system_prompt))
        history.add_message(AIMessage(content="Understood. I will generate Python Plotly code only, using the provided dataframe `df`."))

def create_conversation_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data visualization assistant. Generate Python Plotly code only. Do NOT use markdown or ``` fences. Do NOT call fig.show()."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
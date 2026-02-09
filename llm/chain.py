from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from .model import get_llm

def create_conversation_chain():
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    llm = get_llm()
    return ConversationChain(llm=llm, memory=memory)

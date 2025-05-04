import os
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langsmith import traceable


LANGSMITH_TRACING= True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=os.getenv("LANG_SMITH")
LANGSMITH_PROJECT="document_qa"
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")



# Step 1: Set up environment variables
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY  # 🔐 Replace with your actual API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT


# Step 2: Define a traceable function
@traceable(name="LangSmith Tracing Chat")
def ask_llm(question: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm([HumanMessage(content=question)])
    print("Hello world")
    return response.content



# Step 3: Call your function
if __name__ == "__main__":
    user_question = "Explain the importance of RAG in LLM-based systems."
    answer = ask_llm(user_question)
    print("LLM Response:", answer)
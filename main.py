from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
)
message = [
    SystemMessage(content="Solve the math problem"),
    HumanMessage(content='define set and give an eexample of what is a set and what is not a set')
]

result = llm.invoke(message)
print(result.content)
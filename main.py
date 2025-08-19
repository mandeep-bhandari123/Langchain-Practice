# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
# load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model = "gemini-2.5-flash",
# )
# message = [
#     SystemMessage(content="Solve the math problem"),
#     HumanMessage(content='define set and give an eexample of what is a set and what is not a set')
# ]

# result = llm.invoke(message)
# print(result.content)

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)
template= [("system", " you are an expert on {topic}" ) , ("human" , "tell me the answwer of this {question}")]

prompt_template = ChatPromptTemplate.from_messages(template)

prompt= prompt_template.invoke({"topic":"cats", "question":"Why do cat land on their feet always?"})
print(prompt)
result = llm.invoke(prompt)
print(result.content)
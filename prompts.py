from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)
template= [("system", " you are an expert on cats" ) , ("human" , "tell me the answwer of this {question}")]

prompt_template = ChatPromptTemplate.from_messages(template)

prompt= prompt_template.invoke({"question":"Why do cat land on their feet always?"})

result = llm.invoke(prompt)
print(result.content)
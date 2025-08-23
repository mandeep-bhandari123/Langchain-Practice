from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda
load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash"
                             )

prompt_template 
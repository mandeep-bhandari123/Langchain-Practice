from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda
load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash"
                             )

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a comedian who tells jokes about {topic})"),
     ("human", "Tell me {joke_count} jokes.")]
)
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x : f"World count : {len (x. split ())}\n{x}" )
chain = prompt_template | llm  | StrOutputParser() |uppercase_output | count_words


result  = chain.invoke({"topic":"cats", "joke_count":5})
print(result) 
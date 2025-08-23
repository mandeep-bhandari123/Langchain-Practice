from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda, RunnableParallel
load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash"
                             )

prompt_template  = ChatPromptTemplate.from_messages(
    [
        ("system", " You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}")
    ]
)

def analays_pros(features):
    pros_template= ChatPromptTemplate.from_messages(
    [
        ("system", " You are an expert product reviewer"),
        ("human", "given these {features} , list the pros of these features")
        
    ]
    )
    return pros_template.format_prompt(features = features)



def analays_cons(features):
    cons_template= ChatPromptTemplate.from_messages(
    [
        ("system", " You are an expert product reviewer"),
        ("human", "given these {features} , list the cons of these features")
        
    ]
    )
    return cons_template.format_prompt(features = features)

def combile_pros_cons(pros, cons):
    return f"Pros :\n {pros} , \n \n cons: {cons}"

pros_branch = (
    RunnableLambda (lambda x : analays_pros(x)) | llm | StrOutputParser()
)
cons_branch =(
              RunnableLambda (lambda x : analays_cons(x)) | llm |StrOutputParser()
)

chain = (
    prompt_template 
    | llm
    | StrOutputParser()
    |RunnableParallel(branches = {"pros": pros_branch , "cons":cons_branch})
    | RunnableLambda(lambda x : combile_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name":"MacBook Pro"})

print(result)
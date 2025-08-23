from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda , RunnableBranch
load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

positive_feedback_template= ChatPromptTemplate.from_messages(
    [
        ('system', " You are a healpful assistant"),
        ('human', "Genetate a thank you note for this positive feedback : {feedback}")
        
    ]
)
negetive_feedback_template= ChatPromptTemplate.from_messages(
    [
        ('system', " You are a healpful assistant"),
        ('human', "Genetate responce addressing this negetive feedback : {feedback}")
        
    ]
)

neutral_feedback_template= ChatPromptTemplate.from_messages(
    [
        ('system', " You are a healpful assistant"),
        ('human', "Genetate a request for more detail for this neutral feedback : {feedback}")
        
    ]
)

escalate_feedback_template= ChatPromptTemplate.from_messages(
    [
        ('system', " You are a healpful assistant"),
        ('human', "Genetate a message to escalate this feedbadk to a human agent: {feedback}")
        
    ]
    
)
classification_template= ChatPromptTemplate.from_messages(
    [
        ('system', " You are a healpful assistant"),
        ('human', "classify the sentiment of this feedback as positive , negetive , neutral or escalate : {feedback}")
        
    ]
)

branches = RunnableBranch(
    (
        lambda x : "positive" in x,
        positive_feedback_template | llm | StrOutputParser()
        ),
        (
            lambda x : "negative" in x,
            negetive_feedback_template | llm | StrOutputParser()
        ),
        (
            lambda x : "neutral" in x,
            neutral_feedback_template| llm | StrOutputParser()
        ),
        
            escalate_feedback_template | llm | StrOutputParser()
        )

classification_chain = classification_template | llm | StrOutputParser()

chain = classification_chain | branches

review = "I hated your product , your product broke the moment i bought it. i also tried to return it but i was given a cold sholder by your employee "

result = chain.invoke({"feedback":review})

print(result)
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings , GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = GoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_directory = os.path.join(current_dir , "db", "chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model ='models/embedding-001' )

db = Chroma(persist_directory= persistant_directory,embedding_function=embeddings)

query = "who is Odyessus ?"
retriver = db.as_retriever(
    search_type= "similarity_score_threshold",
    search_kwargs = {"k":5, "score_threshold":0.1}
)

relevant_docs = retriver.invoke(query)

print("\n--- Relevant Documents ---")
for i , doc in enumerate(relevant_docs , 1):
    print(f"Document {i}: \n {doc.page_content} \n")
    if doc.metadata:
        print(f"Source : {doc.metadata.get("source", "unknown")}")
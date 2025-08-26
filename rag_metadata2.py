import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir= os.path.join(current_dir , "db")
persistant_directory= os.path.join(db_dir, "chroma_db_with_metadata")

embedding = GoogleGenerativeAIEmbeddings(model ='models/embedding-001' )

db= Chroma(persist_directory=persistant_directory, 
           embedding_function=embedding)

query = "who is harry?"

retriver = db.as_retriever(
    search_type= "similarity_score_threshold",
    search_kwargs = {"k":1, "score_threshold":0.5}
)

relevant_docs = retriver.invoke(query)

print("Relevant Documents")

for i , doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n {doc.page_content}\n")
    print(f"Source:{doc.metadata['source']}")
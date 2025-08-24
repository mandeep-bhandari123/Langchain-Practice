import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
model = GoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books","odyssey.txt" )
persistant_directory = os.path.join(current_dir , "db", "chroma_db")

if not os.path.exists(persistant_directory):
    print("persistant directory doesnot exist")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File with path {file_path} does not exists")
    
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 0)
    docs = text_splitter.split_documents(documents)
    
    print("\n --- Document Chunks Info ---")
    print(f"Number of document chunks : {len(docs)}")
    print(f"Sample Chunk :\n {docs[0].page_content }\n")
    
    print("\n --- Creating embedding ---")
    embeddings = GoogleGenerativeAIEmbeddings(model ='models/embedding-001' )
    print ("\n --- Finished Creating Vector store --- \n")
    db = Chroma.from_documents(
        docs, embeddings , persist_directory=persistant_directory 
    )
else:
    print("Vector store already exists")

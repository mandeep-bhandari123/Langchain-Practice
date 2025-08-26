import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory :{books_dir}")
print(f"Persistant directory : {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("persistent directory does not exist . Initializing vector store ...")
    
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path "
        )
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
        
    documents = []
    for book_file  in book_files:
        file_path = os.path.join(books_dir , book_file)
        loader = TextLoader(file_path,encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source":book_file}
            documents.append(doc)
    
    text_spliter = CharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100) 
    docs = text_spliter.split_documents(documents)
    
    print("Documents Chunk info")
    print(f"Number of document chunks : {len(docs)}")
    
    
    print("Creating Embedding")
    embedding = GoogleGenerativeAIEmbeddings(model ='models/embedding-001' )
    print("Finished creating embedding")
    print("Creating and persisting vector store")
    db = Chroma.from_documents(
        docs, embedding , persist_directory=persistent_directory
    )

else:
    print("Vector store already exists")
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from dotenv import load_dotenv
import time

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    print("Documents Chunk info")
    print(f"Number of document chunks: {len(docs)}")
    
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    print("Creating and persisting vector store in batches...")
    batch_size = 50
    db = None # Initialize db to None
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        
        while True:
            try:
                print(f"Adding batch {i//batch_size + 1} with {len(batch)} documents...")
                if db is None:
                    # Create the vector store with the first batch
                    db = Chroma.from_documents(
                        batch, 
                        embedding, 
                        persist_directory=persistent_directory
                    )
                else:
                    # Add subsequent batches to the existing vector store
                    db.add_documents(batch)
                
                print(f"Batch {i//batch_size + 1} added successfully.")
                break # Exit the retry loop on success
            except GoogleGenerativeAIError as e:
                if "429 You exceeded your current quota" in str(e):
                    print(f"\n--- Quota Exceeded! Waiting for 90 seconds before retrying... ---")
                    time.sleep(90)
                else:
                    print(f"An unexpected error occurred: {e}")
                    raise # Re-raise other errors
        
        # Add a delay between batches to avoid hitting the quota again
        print("\n--- Waiting for 61 seconds before the next batch ---")
        time.sleep(61)

    print("\nVector store created and persisted successfully.")
    
else:
    print("Vector store already exists.")
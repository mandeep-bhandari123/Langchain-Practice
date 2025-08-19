<<<<<<< HEAD
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
)
message = [
    SystemMessage(content="Solve the math problem"),
    HumanMessage(content='define set and give an eexample of what is a set and what is not a set')
]

result = llm.invoke(message)
print(result.content)
=======
import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import embeddings

# Access the API key from the environment variable
api_key = os.environ.get("GEMINI_API_KEY")



model = GoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

current_dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir,"books","odyssey.tst")
persistent_directory=os.path.join(current_dir,'db','croma_db')

if not os.path.exists(persistent_directory):
    print("Directory doesnot exist")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"the file {file_path} doesnot exist. please check the path."
        )
loader=TextLoader(file_path)
documents = loader.load()
    
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
docs=text_splitter.split_documents(documents)

print("\n--- Document Chunk Information ---")   
print(f"Number of document chunks :{len(docs)}")
print(f"Sample chunk :\n{docs[0].page_content}")

print("\n --- Creting Embeddings ---")

embeddings = embeddings(
    model=""
)    
>>>>>>> ba9c2716490e26f178d6af5f93affe217eb58779

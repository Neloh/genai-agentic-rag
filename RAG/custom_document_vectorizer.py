from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM

load_dotenv()

import logging
logging.basicConfig(filename="../app.log",encoding="utf-8",filemode="a",format="{asctime} - {levelname} - {message}",style="{",datefmt="%Y-%m-%d %H:%M")

loader = PyPDFLoader("../knowledge-base-documents/aws-overview.pdf")
documents = loader.load()

# print("pages in the document/document length: ", len(documents))

text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text",num_ctx=8192)#,model_kwargs={"truncate": True})

from langchain_postgres import PGVector
from langchain_postgres import PGEngine

# 1. Setup the connection string
# Format: postgresql+psycopg://user:password@host:port/dbname

connection = os.environ["DB_URL"]
collection_name = "gen_ai_docs_collection"

BATCH_SIZE = 50 

# 1. Initialize empty store (or connect to existing)
vector_store = PGVector(
    connection=connection,
    embeddings=embeddings,
    collection_name="gen_ai_docs",
    use_jsonb=True
)

# 2. Process in manual batches
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1} (Chunks {i} to {i+len(batch)})...")
    
    try:
        vector_store.add_documents(batch)
    except Exception as e:
        print(f"Error in batch starting at {i}: {e}")

### The below is for testing to see if the ingestion worked! and the responses are based on indexed documents!

# 5. Initialize the Ollama language model
llm = OllamaLLM(model="mistral")

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

query = "What is the main topic of this document? Expand in atleast 3 lines"
result = qa_chain.invoke(query)
print(result)

from cassandra.cluster import Cluster
from langchain_community.vectorstores import Cassandra
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# 1. Establish connection to your local Cassandra 5.0 Docker container
cluster = Cluster(['127.0.0.1']) 
#session = cluster.connect("my_keyspace")
# 2. Define your identifiers (Must match the ones you created in cqlsh)
keyspace = "my_keyspace"
session = cluster.connect(keyspace)
print(f"Connected to {keyspace}!")

table_name = "my_ollama_table"

# 4. Ingest the documents
import pypdf
from langchain_core.documents import Document
from functools import lru_cache

@lru_cache(maxsize=32)
def load_pdf_content_without_metadata(file_path: str) -> list[Document]:
    """
    Loads text from a PDF, excluding pages with no content,
    and returns a list of LangChain Documents without metadata.
    """
    # Open the PDF file
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        total_pages = len(reader.pages)
        content_documents = []

        for i in range(total_pages):
            page = reader.pages[i]
            # Extract text content from the page
            page_text = page.extract_text()

            # Check if the page has significant content (not just whitespace)
            if page_text and page_text.strip():
                # Create a LangChain Document with only page_content and an empty metadata dict
                # LangChain Document class definition: Document(page_content: str, metadata: dict)
                doc = Document(page_content=page_text.strip(), metadata={})
                content_documents.append(doc)

    return content_documents

# Example usage:
file_path = "../knowledge-base-documents/sagemaker-dg.pdf"
documents = load_pdf_content_without_metadata(file_path)

# print("pages in the document/document length: ", len(documents))
'''
#text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False)
text_splitter = CharacterTextSplitter(
    separator="\n",  # Add this!
    chunk_size=800, 
    chunk_overlap=80, 
    length_function=len, 
    is_separator_regex=False
)
texts = text_splitter.split_documents(documents)
'''
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Smarter splitting (prevents the 3701 size warning)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    separators=["\n\n", "\n", " ", ""]
)
texts = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text",num_ctx=8192)#,model_kwargs={"truncate": True})

vstore = Cassandra(
    embedding=embeddings,
    session=session,
    keyspace=keyspace,
    table_name=table_name,
    setup_mode="off",
    #vector_dimension=768
)
# Since your 'texts' chunks already have metadata={}, no metadata will be stored
#ids = vstore.add_documents(texts)
#print(f"Successfully ingested {len(ids)} chunks into {keyspace}.{table_name}")

print(f"Original Documents: {len(documents)}")
print(f"Chunks created: {len(texts)}")
BATCH_SIZE=100
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1} (Chunks {i} to {i+len(batch)})...")
    if len(batch) > 0:
        print(f"First chunk preview: {texts[0].page_content[:50]}")
        ids = vstore.add_documents(batch)
    else:
        print("Error: No chunks were created. Check your splitter settings.")

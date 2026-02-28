from langchain_neo4j import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
#import os
from langchain_community.document_loaders import PyPDFLoader

import logging
logging.basicConfig(filename="../app.log",encoding="utf-8",filemode="a",format="{asctime} - {levelname} - {message}",style="{",datefmt="%Y-%m-%d %H:%M")

from langchain_huggingface import HuggingFaceEmbeddings

# This runs locally and does NOT need an API key
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# 1. Load and split your document
'''
loader = PyPDFLoader("../knowledge-base-documents/sagemaker-dg.pdf")
documents = loader.load()
# print("pages in the document/document length: ", len(documents))
'''
import pypdf
from langchain_core.documents import Document

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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=80
)

texts = text_splitter.split_documents(documents)

# 2. Initialize Neo4j as a Vector Store
# This creates the nodes and the vector index automatically
# 2. Process in manual batches
BATCH_SIZE=100
neo4j_db = None

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1} (Chunks {i} to {i+len(batch)})...")
    if neo4j_db is None:
        neo4j_db = Neo4jVector.from_documents(
                batch,
                local_embeddings,
                url="bolt://localhost:7687",
                username="neo4j",
                password="Qwe123321#",
                index_name="sagemaker_vector_index", # Name for your vector index
                node_label="Chunk"           # Label for the nodes in the graph
                )
    else:
        neo4j_db.add_documents(batch)
print(f"Ingestion complete!")

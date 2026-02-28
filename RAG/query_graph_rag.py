from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings

# Must use the EXACT same model and device as ingestion
local_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Connect to the existing index
existing_db = Neo4jVector.from_existing_index(
    local_embeddings,
    url="bolt://localhost:7687",
    username="neo4j",
    password="Qwe123321#",
    index_name="sagemaker_vector_index",
)

# Run a test query
#query = "How do I use SageMaker for model training?"
#query = "How many IP addresses will be used for a training or inference job in SageMaker?"
query = "How many IP addresses will be used for a training or inference job in SageMaker? Think of a VpcConfig and how it works? Look for actual page that explains the number of ip addresses in a training or sagemaker inference endpoint"
results = existing_db.similarity_search(query, k=3)

for i, res in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(res.page_content)

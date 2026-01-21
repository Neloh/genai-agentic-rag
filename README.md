# genai-agentic-rag
Building an end-to-end agentic rag workflow with aim of understanding AWS Documentation. 

## Objectives:
- Learn how LangChain SDK works
- Learn how to make use Ollama free-models for Retrieval Augmented Generation project and also will consider making it AgenticRAG/Advanced RAG
- Use PostgresQL as a Vector database
- Monitor end-to-end of this workflow by enabling PgAdmin4 for database monitoring
  - construct metrics for the CPU and GPU utilization
  - Introduce AI observability (OpenTelemetry) in order to track for any AI related issues.
- MCPify the project, to give the Large language models more contexts (adding tools via Model Context Protocol )
- Possibly write a frontend in Streamlit that will show a document upload, how it gets ingested and then used by clients to query but also making sure the agents can have a some sort of memory.

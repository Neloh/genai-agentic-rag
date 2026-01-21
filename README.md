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

### Indexing the documents works
![](https://github.com/Neloh/genai-agentic-rag/blob/main/images/langchain_embedding_vector_ingested_in_postgres.png)

- The RAG part of this project is similar to what the article shows just that this is running locally in a Debian machine (running on Ubuntu 24.04.03)
- Hardware limitations:
    ```
    GPU: NVIDIA GeForce 940MX (likely a laptop GPU).
    Driver: NVIDIA Driver 580.95.05 (very recent version for 2026).
    CUDA: CUDA Version 13.0 is supported by this driver.
    ```
### GPU metrics function tool shows activity during ingestion:

![](https://github.com/Neloh/genai-agentic-rag/blob/main/images/gpu_metrics_tool_works_traffic_increase.png)

## WORK IN PROGRESS
- To investigate more on AgenticRAG, however adopting the same approach as [Langchaing Agents SDK here](https://docs.langchain.com/oss/python/langchain/agents), I can get to the throttle due to not having a proper OpenAI_API_KEY. Ollama works locally in my machine and the project can be flipped over to production (by setting DEV=`False` in [agents_with_tools.py](https://github.com/Neloh/genai-agentic-rag/blob/main/agents/agents_with_tools.py) script). 
```
openai.RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}
```
- (ambitions!) Might add Prometheus and Grafana for Observability (unless there is a Python based library like [ADOT](https://docs.aws.amazon.com/xray/latest/devguide/xray-services-adot.html) that can handle adding observability to the project )
- Integrated with Streamlit or Dash library in Python for a Frontend.

# Graph RAG System with Neo4j Vector Index

A production-style **Retrieval-Augmented Generation (RAG)** implementation using:

- Neo4j (Vector Index + Graph relationships)
- Sentence Transformers (384-dim embeddings)
- LangChain
- Batch ingestion & scalable embedding pipelines
- Graph-augmented retrieval

This project demonstrates practical, systems-level understanding of modern RAG architecture — from ingestion to semantic retrieval — including real-world debugging of vector index mismatches, batching strategies, and transformer context constraints.

---

## 🎯 Objective

Build a scalable, graph-aware RAG system that:

- Ingests large documents (20k+ chunks)
- Generates embeddings efficiently
- Stores vectors in Neo4j
- Enables similarity search
- Enhances retrieval using graph relationships
- Supports structured traversal beyond naive vector search

---

## 🧠 System Architecture

Document → Chunking → Embeddings → Neo4j Vector Index  
                                         ↓  
                                    Graph Relationships  
                                         ↓  
                                  Semantic Retrieval  

### Core Design Decisions

**Embedding Model**
- `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional vectors
- Cosine similarity

**Graph Model**
- `(:Document)`
- `(:Chunk)`
- `(:Chunk)-[:PART_OF]->(:Document)`
- `(:Chunk)-[:NEXT]->(:Chunk)`

This enables:
- Vector similarity search
- Sequential context expansion
- Document-level filtering

---

## 📊 Scale Achieved

- 20,653 chunks ingested
- Batch embedding pipeline (100 chunks per batch)
- Online vector index population
- Verified similarity search queries

---

## ⚙️ Vector Index Configuration

```cypher
CREATE VECTOR INDEX sagemaker_vector_index IF NOT EXISTS
FOR (n:Chunk)
ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
};


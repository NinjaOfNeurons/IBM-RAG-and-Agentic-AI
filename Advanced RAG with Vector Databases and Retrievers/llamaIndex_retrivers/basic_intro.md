# LangChain and LlamaIndex Retrievers Overview

## LangChain Retrievers

A **LangChain retriever** is an interface that returns documents based on an **unstructured query**.

There are several types of LangChain retrievers.

---

## Vector Store-Based Retriever

A **vector store-based retriever** retrieves documents from a **vector database**.

A vector store-based retriever can be created directly from the vector store object using the `retriever` method.

Two main retrieval strategies:

### Similarity Search
- The retriever accepts a query.
- It retrieves the **most semantically similar data**.

### Maximum Marginal Relevance (MMR)
- Balances **relevance** and **diversity** of retrieved results.
- Helps avoid returning very similar documents.

---

## Multi-Query Retriever

The **multi-query retriever** uses an LLM to:
- Generate **multiple variations of the original query**
- Retrieve a **richer set of documents**

---

## Self-Query Retriever

The **self-query retriever** converts the query into two components:

1. **Semantic query string** — used for vector similarity search  
2. **Metadata filter** — applied alongside the semantic search

---

## Parent Document Retriever

The **parent document retriever** uses **two text splitters**:

- **Parent splitter**
  - Splits text into **large chunks**
  - These chunks are retrieved

- **Child splitter**
  - Splits documents into **smaller chunks**
  - Used to generate **more meaningful embeddings**

---

# LlamaIndex Index Types

The core **LlamaIndex index types** are:

- **VectorStoreIndex**
- **DocumentSummaryIndex**
- **KeywordTableIndex**

---

## VectorStoreIndex

The **VectorStoreIndex**:

- Stores **vector embeddings** for each document chunk
- Best suited for **semantic retrieval**
- Commonly used in **LLM pipelines**

---

## DocumentSummaryIndex

The **DocumentSummaryIndex**:

- Generates and stores **summaries of documents**
- Summaries are used to **filter documents before retrieving full content**
- Useful when working with **large and diverse document sets**

---

## KeywordTableIndex

The **KeywordTableIndex**:

- Extracts **keywords from documents**
- Maps keywords to **specific content chunks**
- Useful in **hybrid or rule-based search scenarios**

---

# LlamaIndex Retrievers

## Vector Index Retriever

The **Vector Index Retriever**:

- Uses **vector embeddings**
- Finds **semantically related content**
- Ideal for **general-purpose search and RAG pipelines**

---

## BM25 Retriever

The **BM25 Retriever**:

- Uses a **keyword-based ranking method**
- Retrieves documents based on **exact keyword matches**
- Does **not rely on semantic similarity**

---

## Document Summary Index Retriever

The **Document Summary Index Retriever**:

- Uses **document summaries** instead of full documents
- Helps find relevant content more efficiently

Two versions exist:

- **LLM-based summary retriever**
- **Semantic similarity-based summary retriever**

---

## Auto Merging Retriever

The **Auto Merging Retriever**:

- Preserves context in **long documents**
- Uses **hierarchical chunking**
- Breaks documents into **parent and child nodes**

---

## Recursive Retriever

The **Recursive Retriever**:

- Follows **relationships between nodes**
- Uses references such as:
  - Citations in academic papers
  - Metadata links

---

## Query Fusion Retriever

The **Query Fusion Retriever**:

- Combines results from **multiple retrievers**
- Uses **fusion strategies** to merge results

### Supported Fusion Strategies

- **Reciprocal Rank Fusion**
- **Relative Score Fusion**
- **Distribution-Based Fusion**
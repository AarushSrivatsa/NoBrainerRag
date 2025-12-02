[![PyPI version](https://img.shields.io/pypi/v/NoBrainerRag.svg)](https://pypi.org/project/NoBrainerRag/)

# NoBrainer RAG

A dead simple RAG (Retrieval-Augmented Generation) system that just works. Built for developers who want to add memory to their AI applications without overthinking it.

## Why NoBrainer?

- 🚀 **Simple API** - Just 3 methods: add(), query(), clear()
- 🔒 **Namespace Isolation** - Each conversation gets its own isolated namespace
- 💾 **Persistent Memory** - Data survives even if your object doesn't
- 🎯 **Smart Retrieval** - RecursiveCharacterTextSplitter + FlashRank reranking built-in
- ⚡ **Flexible** - Swap embedding models, toggle reranking, adjust on the fly
- 🌍 **Cloud Agnostic** - Works with AWS, GCP, or Azure Pinecone regions

## Installation

```bash
pip install NoBrainerRag
```

Or clone and install:

```bash
git clone https://github.com/AarushSrivatsa/NoBrainerRag.git
cd NoBrainerRag
pip install -e .
```

---

## What Makes This Actually Good

Most RAG tutorials give you basic vector search and call it a day. NoBrainer comes with **production-grade features out of the box**:

### 📊 RecursiveCharacterTextSplitter (Smart Chunking)

Not your average "split every N characters" nonsense. This intelligently splits text by:
1. Paragraphs first (`\n\n`)
2. Then sentences (`.`)
3. Then clauses (`,`)
4. Finally words as a last resort

**Why this matters:** Preserves semantic meaning. You don't get chunks that cut off mid-sentence or split important context.

### 🎯 FlashRank Reranking (Better Results)

Every single retrieval automatically:
1. Gets 10 candidate chunks from Pinecone
2. Runs them through FlashRank (`ms-marco-MiniLM-L-12-v2`)
3. Returns only the top 4 most relevant

**Why this matters:** Vector similarity isn't perfect. Reranking catches what embeddings miss, giving you the *actually* relevant results.

**You can toggle this on/off anytime** - just set `rag.use_reranking = False` if you want raw vector search.

### 🧠 Flexible Embedding Models

Defaults to `nomic-embed-text:v1.5` via Ollama - one of the best open-source embedding models. Runs locally.

**But you can use ANY embedding model:**
- OpenAI embeddings
- Cohere embeddings
- HuggingFace models
- Any LangChain-compatible embedding

**Why this matters:** No vendor lock-in. Use what works best for your use case.

---

## Prerequisites

### 1. Pinecone API Key (Required)

Get your free API key from [Pinecone](https://www.pinecone.io/). 

Create a `.env` file:

```env
PINECONE_API_KEY=your_key_here
```

### 2. Ollama with Embedding Model (Only if using default embeddings)

If you're using the default Ollama embeddings, install [Ollama](https://ollama.ai/) and pull the model:

```bash
ollama pull nomic-embed-text:v1.5
```

If you're bringing your own embedding model, skip this step.

---

## Quick Start

```python
from NoBrainerRag import NoBrainerRag

# Create a RAG instance with a namespace
rag = NoBrainerRag(
    namespace="user_123",
    index_name="my-rag-index"
)

# Insert some knowledge
rag.add("Paris is the capital of France.")
rag.add("Python is a programming language created by Guido van Rossum.")

# Retrieve relevant info
result = rag.query("What is the capital of France?")
print(result)

# Delete when done
rag.clear()
```

---

## 🔥 The Magic: Persistent Memory

Here's the cool part - **your data persists even if the object is gone**:

```python
# Session 1: Insert data
rag = NoBrainerRag(namespace="user_123", index_name="my-index")
rag.add("Important information here")
del rag  # Object is destroyed

# Session 2: Access the same data later
rag = NoBrainerRag(namespace="user_123", index_name="my-index")  # Same namespace!
result = rag.query("tell me about important information")
# Your data is still there! 🎉
```

### 🗂️ How Pinecone Storage Works

**Index = The Database**  
Your `index_name` is the top-level Pinecone index where ALL your data lives. Think of it as your database.

**Namespace = The Conversation**  
Your `namespace` parameter creates an isolated namespace INSIDE that index. Each conversation is completely isolated.

```
Pinecone Index: "my-chatbot-memory"
├── Namespace: "user_123" (namespace="user_123")
│   ├── chunk_1: "Paris is the capital..."
│   ├── chunk_2: "Python was created..."
│   └── chunk_3: "Machine learning is..."
├── Namespace: "user_456" (namespace="user_456")
│   ├── chunk_1: "Tokyo is in Japan..."
│   └── chunk_2: "JavaScript runs in..."
└── Namespace: "doc_789" (namespace="doc_789")
    └── chunk_1: "This document explains..."
```

**What this means:**
- One Pinecone index can hold thousands of namespaces
- Each namespace is completely isolated (no data leakage)
- Same index + same namespace = same memory, always
- Delete a namespace = only that data is wiped

### ⚠️ CRITICAL: How Memory Persistence Works

To access the **exact same memory** across sessions, you MUST have:

✅ **Same Pinecone index** (`index_name` parameter)  
✅ **Same namespace** (`namespace` parameter)

```python
# These will access THE SAME memory:
rag1 = NoBrainerRag(namespace="user_123", index_name="my-index")
rag2 = NoBrainerRag(namespace="user_123", index_name="my-index")
# ✅ Same data

# These will have DIFFERENT memory:
rag1 = NoBrainerRag(namespace="user_123", index_name="my-index")
rag2 = NoBrainerRag(namespace="user_456", index_name="my-index")
# ❌ Different namespace = different memory

# These will also have DIFFERENT memory:
rag1 = NoBrainerRag(namespace="user_123", index_name="my-index")
rag2 = NoBrainerRag(namespace="user_123", index_name="other-index")
# ❌ Different index = completely different database = different memory
```

**The Rule:** Same index + same namespace = same memory. Change either and you get a fresh memory space.

---

## API Reference

### Initialization

```python
rag = NoBrainerRag(
    namespace="user_123",                   # Required: Unique namespace identifier
    index_name="my-rag-index",              # Required: Pinecone index name
    embedding_model=None,                   # Optional: Custom embedding model
    chunk_size=400,                         # Optional: Size of text chunks
    chunk_overlap=75,                       # Optional: Overlap between chunks
    separators=["\n\n", "\n", ".", ",", " ", ""],  # Optional: Split points
    base_k=10,                              # Optional: Initial retrieval count
    top_n=4,                                # Optional: Results after reranking
    use_reranking=True,                     # Optional: Enable FlashRank
    rerank_model="ms-marco-MiniLM-L-12-v2", # Optional: Reranking model
    pinecone_cloud="aws",                   # Optional: Cloud provider
    pinecone_region="us-east-1",            # Optional: Region
    similarity_metric="cosine"              # Optional: Vector similarity metric
)
```

### Methods

**add(text: str)**  
Insert text into the vector database. Automatically chunks and embeds it.

```python
rag.add("Your text here")
# Returns: "Insertion Successful: 3 chunks created"
```

**query(query: str)**  
Retrieve relevant content for a query.

```python
results = rag.query("What is the capital of France?")
# Returns formatted string with top relevant chunks
```

**clear()**  
Delete all documents in this namespace.

```python
rag.clear()
# Returns: "RAG memory of namespace 'user_123' was successfully wiped out"
```

---

## Advanced Usage

### Custom Embedding Models

```python
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

# Use OpenAI embeddings
rag = NoBrainerRag(
    namespace="user_123",
    index_name="my-index",
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Or Cohere
rag = NoBrainerRag(
    namespace="user_123",
    index_name="my-index",
    embedding_model=CohereEmbeddings(model="embed-english-v3.0")
)
```

### Disable Reranking for Speed

```python
# At initialization
rag = NoBrainerRag(
    namespace="user_123",
    index_name="my-index",
    use_reranking=False  # Skip reranking for faster results
)

# Or toggle it anytime
rag.use_reranking = False
result = rag.query("fast query")  # Uses raw vector search

rag.use_reranking = True
result = rag.query("precise query")  # Uses reranking
```

### Adjust Retrieval Parameters on the Fly

```python
rag = NoBrainerRag(namespace="user_123", index_name="my-index")

# Start with defaults (base_k=10, top_n=4)
result = rag.query("my query")

# Need more context? Change it
rag.base_k = 20
rag.top_n = 8
result = rag.query("complex query")  # Now retrieves more chunks

# Back to focused results
rag.base_k = 5
rag.top_n = 2
result = rag.query("simple query")
```

### Multi-Region Setup

```python
# Use GCP in Europe
rag = NoBrainerRag(
    namespace="user_123",
    index_name="my-index",
    pinecone_cloud="gcp",
    pinecone_region="europe-west1"
)

# Or Azure in East US
rag = NoBrainerRag(
    namespace="user_123",
    index_name="my-index",
    pinecone_cloud="azure",
    pinecone_region="eastus"
)
```

### Custom Chunking Strategy

```python
# Larger chunks with more overlap
rag = NoBrainerRag(
    namespace="user_123",
    index_name="my-index",
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n\n", "\n\n", "\n"]  # Only split on paragraph breaks
)
```

---

## Common Use Cases

### Chatbot with Memory

```python
# When user starts chatting
rag = NoBrainerRag(namespace=user.id, index_name="chatbot-memory")

# As conversation progresses
rag.add(f"User said: {user_message}")
rag.add(f"Assistant replied: {bot_response}")

# When generating responses
context = rag.query(user_message)
# Feed context to your LLM
```

### Document Q&A

```python
rag = NoBrainerRag(namespace="doc_session_456", index_name="documents")

# Load your document
with open("document.txt") as f:
    content = f.read()
    rag.add(content)

# Ask questions
answer = rag.query("What is the main topic?")
```

### Multi-User Application

```python
# Each user gets isolated memory (separate namespace)
user1_rag = NoBrainerRag(namespace=f"user_{user1.id}", index_name="app-memory")
user2_rag = NoBrainerRag(namespace=f"user_{user2.id}", index_name="app-memory")

# Their data never mixes - guaranteed namespace isolation
```

---

## Under the Hood

NoBrainer RAG uses **battle-tested, production-grade tools** so you don't have to piece them together yourself:

- **Embeddings**: Ollama with `nomic-embed-text:v1.5` (768 dimensions, state-of-the-art, runs locally)
- **Chunking**: LangChain's `RecursiveCharacterTextSplitter` - respects semantic boundaries
- **Reranking**: FlashRank with `ms-marco-MiniLM-L-12-v2` - automatically improves precision
- **Vector Database**: Pinecone (serverless, production-scale)
- **Retrieval**: LangChain's contextual compression retriever (retrieves 10 → reranks → returns top 4)

**The pipeline:**
1. **Text** → RecursiveCharacterTextSplitter breaks it into semantic chunks
2. **Chunks** → Nomic embeddings convert to 768-dim vectors
3. **Store** → Pinecone index with namespace isolation
4. **Query** → Retrieve top 10 candidates based on vector similarity
5. **Rerank** → FlashRank re-scores all 10 and picks the actual best 4 matches
6. **Return** → Formatted, contextually relevant results ready to use

This isn't a toy setup - **this is the right way to do RAG.** The kind of pipeline you'd spend a week researching and building yourself. Except it's already done.

---

## FAQ

**Q: Do I need to keep the same NoBrainerRag object alive?**  
A: Nope! As long as you use the same index name and namespace, you can create new objects anytime and access the same data.

**Q: What happens if I use the same namespace twice?**  
A: That's the point! Same index + same namespace = same memory. It's a feature, not a bug.

**Q: Can I use this in production?**  
A: Yeah, it's built on production-grade tools (Pinecone, LangChain, Ollama). Just make sure your Pinecone plan can handle your scale.

**Q: How much does Pinecone cost?**  
A: They have a generous free tier. Check [Pinecone pricing](https://www.pinecone.io/pricing/).

**Q: Can I change the embedding model?**  
A: Yes! Pass any LangChain-compatible embedding model to the `embedding_model` parameter.

**Q: Is my data secure?**  
A: Data is stored in your Pinecone account. Use their security features + keep your API keys safe.

**Q: Can I adjust retrieval settings after initialization?**  
A: Yes! Just change instance variables like `rag.base_k = 20` or `rag.use_reranking = False` and the next query will use the new settings.

**Q: What's the difference between index and namespace?**  
A: Index = your database. Namespace = an isolated partition inside that database. One index can hold many namespaces.

---

## Requirements

- Python 3.8+
- Pinecone API key
- Ollama installed locally (only if using default embeddings)
- `nomic-embed-text:v1.5` model pulled in Ollama (only if using default embeddings)

---

## Contributing

Found a bug? Have an idea? PRs welcome! Keep it simple though - the goal is "no brainer", not "all the features".

---

## License

MIT - do whatever you want with it.

---

## Support

If this saved you hours of work, star the repo ⭐ and help other devs find it!

---

**Built with ❤️ for developers who just want things to work.**
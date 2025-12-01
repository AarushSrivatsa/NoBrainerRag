[![PyPI version](https://img.shields.io/pypi/v/NoBrainerRag.svg)](https://pypi.org/project/NoBrainerRag/)

# NoBrainer RAG

A dead simple RAG (Retrieval-Augmented Generation) system that just works. Built for developers who want to add memory to their chatbots without overthinking it.

## Why NoBrainer?

- 🚀 **Simple API** - Just 3 methods: insert, retrieve, delete
- 🔒 **Conversation Isolation** - Each conversation gets its own namespace
- 💾 **Persistent Memory** - Data survives even if your object doesn't
- 🎯 **Smart Retrieval** - RecursiveCharacterTextSplitter + FlashRank reranking built-in (no configuration needed)
- ⚡ **Fast Setup** - Get running in under 5 minutes

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

### 🧠 Nomic Embeddings (No API Costs)

Uses `nomic-embed-text:v1.5` - one of the best open-source embedding models. Runs locally via Ollama.

**Why this matters:** No OpenAI bills. No rate limits. No data leaving your machine. And the quality is genuinely competitive with paid options.

**You get all of this by default.** Zero configuration. Just install and go.

---

## Prerequisites

### 1. Pinecone API Key and Index Name (Required)

Get your free API key from [Pinecone](https://www.pinecone.io/). 

Create a `.env` file with **BOTH** of these:

```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=your-index-name-here
```

> ⚠️ **IMPORTANT**: 
> - Both the API key AND index name are required
> - Index name MUST be lowercase with hyphens (e.g., `my-rag-index`, not `My_RAG_Index`)
> - The library will auto-create the index if it doesn't exist (768 dimensions, cosine similarity, AWS us-east-1)

### 2. Ollama with Embedding Model (Required)

Install [Ollama](https://ollama.ai/) and pull the embedding model:

```bash
ollama pull nomic-embed-text:v1.5
```

---

## Quick Start

```python
from NoBrainerRag import NoBrainerRag

# Create a RAG instance for a conversation
rag = NoBrainerRag(convo_id="user_123")

# Insert some knowledge
rag.insertIntoVectorDB("Paris is the capital of France.")
rag.insertIntoVectorDB("Python is a programming language created by Guido van Rossum.")

# Retrieve relevant info
result = rag.retrieveFromVectorDB("What is the capital of France?")
print(result)

# Delete when done
rag.deleteConvoDB()
```

---

## 🔥 The Magic: Persistent Memory

Here's the cool part - **your data persists even if the object is gone**:

```python
# Session 1: Insert data
rag = NoBrainerRag(convo_id="user_123")
rag.insertIntoVectorDB("Important information here")
del rag  # Object is destroyed

# Session 2: Access the same data later
rag = NoBrainerRag(convo_id="user_123")  # Same ID!
result = rag.retrieveFromVectorDB("tell me about important information")
# Your data is still there! 🎉
```

### 🗂️ How Pinecone Storage Works

**Index = The Database**  
Your `PINECONE_INDEX_NAME` is the top-level Pinecone index where ALL your data lives. Think of it as your database.

**Namespace = The Conversation**  
Your `convo_id` becomes a namespace INSIDE that index. Each conversation is isolated in its own namespace.

```
Pinecone Index: "my-chatbot-memory"
├── Namespace: "user_123" (convo_id="user_123")
│   ├── chunk_1: "Paris is the capital..."
│   ├── chunk_2: "Python was created..."
│   └── chunk_3: "Machine learning is..."
├── Namespace: "user_456" (convo_id="user_456")
│   ├── chunk_1: "Tokyo is in Japan..."
│   └── chunk_2: "JavaScript runs in..."
└── Namespace: "doc_789" (convo_id="doc_789")
    └── chunk_1: "This document explains..."
```

**What this means:**
- One Pinecone index can hold thousands of conversations
- Each `convo_id` is completely isolated (no data leakage)
- Same index + same `convo_id` = same memory, always
- Delete a namespace = only that conversation is wiped

### ⚠️ CRITICAL: How Memory Persistence Works

To access the **exact same memory** across sessions, you MUST have:

✅ **Same Pinecone index** (`PINECONE_INDEX_NAME` in `.env`)  
✅ **Same conversation ID** (`convo_id` parameter)

```python
# These will access THE SAME memory:
rag1 = NoBrainerRag(convo_id="user_123")  # index: "my-index"
rag2 = NoBrainerRag(convo_id="user_123")  # index: "my-index"
# ✅ Same data

# These will have DIFFERENT memory:
rag1 = NoBrainerRag(convo_id="user_123")  # index: "my-index"
rag2 = NoBrainerRag(convo_id="user_456")  # index: "my-index"
# ❌ Different convo_id = different namespace = different memory

# These will also have DIFFERENT memory:
rag1 = NoBrainerRag(convo_id="user_123")  # index: "my-index"
rag2 = NoBrainerRag(convo_id="user_123")  # index: "other-index"
# ❌ Different index = completely different database = different memory
```

**The Rule:** Same index + same convo_id = same memory. Change either and you get a fresh memory space.

**This means you can:**
- Restart your app without losing conversation history
- Share conversation data across different servers/processes
- Implement true long-term memory for your chatbots
- Resume conversations after crashes, deploys, or anything else

---

## Usage

### Insert Text

```python
rag.insertIntoVectorDB("Your text here")
# Returns: "Insertion Successful 3 chunks created"
```

### Retrieve Relevant Content

```python
results = rag.retrieveFromVectorDB("Your query")
# Returns formatted string with the top relevant chunks:
# ---DOCUMENT 1---
# [content]
# ---END OF DOCUMENT 1---
```

### Delete Conversation

```python
rag.deleteConvoDB()
# Returns: "Rag Memory of convo with user_123 id was successfully wiped out"
```

---

## Configuration

Customize the behavior when initializing:

```python
rag = NoBrainerRag(
    convo_id="user_123",
    chunk_size=400,           # Size of each text chunk
    chunk_overlap=75,         # Overlap between chunks
    separators=["\n\n", "\n", ".", ",", " ", ""],  # How to split text
    base_k=10,                # Initial retrieval count
    top_n=4                   # Final results after reranking
)
```

### Parameters Explained

- **convo_id**: Unique identifier for the conversation (string or int) - becomes a Pinecone namespace
- **chunk_size**: How many characters per chunk (default: 400)
- **chunk_overlap**: Character overlap between chunks for context (default: 75)
- **separators**: Preferred split points, in order of priority (default: paragraphs → lines → sentences → words)
- **base_k**: How many chunks to retrieve initially (default: 10)
- **top_n**: How many chunks to return after reranking (default: 4)

---

## Under the Hood

NoBrainer RAG uses **battle-tested, production-grade tools** so you don't have to piece them together yourself:

- **Embeddings**: Ollama with `nomic-embed-text:v1.5` (768 dimensions, state-of-the-art, runs locally)
- **Chunking**: LangChain's `RecursiveCharacterTextSplitter` - respects semantic boundaries instead of dumb character splits
- **Reranking**: FlashRank with `ms-marco-MiniLM-L-12-v2` - automatically improves precision on every query
- **Vector Database**: Pinecone (serverless, AWS us-east-1, production-scale)
- **Retrieval**: LangChain's contextual compression retriever (retrieves 10 → reranks → returns top 4)

**The pipeline:**
1. **Text** → RecursiveCharacterTextSplitter breaks it into semantic chunks (respects paragraphs, sentences)
2. **Chunks** → Nomic embeddings convert to 768-dim vectors (no API costs, runs local)
3. **Store** → Pinecone index with namespace isolation per `convo_id`
4. **Query** → Retrieve top 10 candidates based on vector similarity
5. **Rerank** → FlashRank re-scores all 10 and picks the actual best 4 matches
6. **Return** → Formatted, contextually relevant results ready to use

This isn't a toy setup - **this is the right way to do RAG.** The kind of pipeline you'd spend a week researching and building yourself. Except it's already done.

---

## Common Use Cases

### Chatbot with Memory

```python
# When user starts chatting
rag = NoBrainerRag(convo_id=user.id)

# As conversation progresses
rag.insertIntoVectorDB(f"User said: {user_message}")
rag.insertIntoVectorDB(f"Assistant replied: {bot_response}")

# When generating responses
context = rag.retrieveFromVectorDB(user_message)
# Feed context to your LLM
```

### Document Q&A

```python
rag = NoBrainerRag(convo_id="doc_session_456")

# Load your document
with open("document.txt") as f:
    content = f.read()
    rag.insertIntoVectorDB(content)

# Ask questions
answer = rag.retrieveFromVectorDB("What is the main topic?")
```

### Multi-User Application

```python
# Each user gets isolated memory (separate namespace)
user1_rag = NoBrainerRag(convo_id=f"user_{user1.id}")
user2_rag = NoBrainerRag(convo_id=f"user_{user2.id}")

# Their data never mixes - guaranteed namespace isolation
```

---

## FAQ

**Q: Do I need to keep the same NoBrainerRag object alive?**  
A: Nope! As long as you use the same index name and convo_id, you can create new objects anytime and access the same data.

**Q: What happens if I use the same convo_id twice?**  
A: That's the point! Same index + same ID = same memory. It's a feature, not a bug.

**Q: Can I use this in production?**  
A: Yeah, it's built on production-grade tools (Pinecone, LangChain, Ollama). Just make sure your Pinecone plan can handle your scale.

**Q: How much does Pinecone cost?**  
A: They have a generous free tier. Check [Pinecone pricing](https://www.pinecone.io/pricing/).

**Q: Can I change the embedding model?**  
A: Currently it uses `nomic-embed-text:v1.5`. Fork it if you need something different.

**Q: Is my data secure?**  
A: Data is stored in your Pinecone account. Use their security features + keep your API keys safe.

**Q: Why does the index name matter so much?**  
A: The index is the top-level container (the database) for ALL your data. Conversations live as namespaces inside it. Different index = completely different data store.

**Q: What if my index name has uppercase letters?**  
A: Pinecone doesn't allow uppercase in index names. Use lowercase with hyphens only (e.g., `my-rag-index`). The library will throw an error if you try to use uppercase.

**Q: What's the difference between index and namespace?**  
A: Index = your database. Namespace = a conversation inside that database. One index can hold many namespaces (conversations).

---

## Requirements

- Python 3.8+
- Pinecone API key **and index name** (lowercase with hyphens only)
- Ollama installed locally
- `nomic-embed-text:v1.5` model pulled in Ollama

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
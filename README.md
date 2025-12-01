# NoBrainer RAG

A dead simple RAG (Retrieval-Augmented Generation) system that just works. Built for developers who want to add memory to their chatbots without overthinking it.

## Why NoBrainer?

- 🚀 **Simple API** - Just 3 methods: insert, retrieve, delete
- 🔒 **Conversation Isolation** - Each conversation gets its own namespace
- 💾 **Persistent Memory** - Data survives even if your object doesn't
- 🎯 **Smart Retrieval** - Recursive text splitting + built-in reranking for accurate results
- ⚡ **Fast Setup** - Get running in under 5 minutes

## Installation

```bash
pip install nobrainer-rag
```

Or clone and install:
```bash
git clone https://github.com/yourusername/nobrainer-rag.git
cd nobrainer-rag
pip install -e .
```

## Prerequisites

### 1. Pinecone API Key (Required)

Get your free API key from [Pinecone](https://www.pinecone.io/). Create a `.env` file:

```env
PINECONE_API_KEY=your_key_here
```

### 2. Ollama with Embedding Model (Required)

Install [Ollama](https://ollama.ai/) and pull the embedding model:

```bash
ollama pull nomic-embed-text:v1.5
```

## Quick Start

```python
from nobrainer_rag import NoBrainerRag

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

## The Magic: Persistent Memory

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

This means you can:
- Restart your app without losing conversation history
- Share conversation data across different parts of your application
- Implement true long-term memory for your chatbots

## Usage

### Insert Text

```python
rag.insertIntoVectorDB("Your text here")
# Returns: "Insertion Successful 3 chunks created"
```

### Retrieve Relevant Content

```python
results = rag.retrieveFromVectorDB("Your query")
# Returns formatted string with the top relevant chunks
```

### Delete Conversation

```python
rag.deleteConvoDB()
# Returns: "Rag Memory of convo with user_123 id was successfully wiped out"
```

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

- **convo_id**: Unique identifier for the conversation (string or int)
- **chunk_size**: How many characters per chunk (default: 400)
- **chunk_overlap**: Character overlap between chunks for context (default: 75)
- **separators**: Preferred split points, in order of priority (default: paragraphs > lines > sentences > words)
- **base_k**: How many chunks to retrieve initially (default: 10)
- **top_n**: How many chunks to return after reranking (default: 4)

## Under the Hood

NoBrainer RAG uses battle-tested tools so you don't have to:

- **Embeddings**: [Ollama](https://ollama.ai/) with `nomic-embed-text:v1.5` (768 dimensions)
- **Vector Database**: [Pinecone](https://www.pinecone.io/) (serverless, AWS us-east-1)
- **Chunking**: LangChain's `RecursiveCharacterTextSplitter`
- **Reranking**: [Flashrank](https://github.com/PrithivirajDamodaran/FlashRank) with `ms-marco-MiniLM-L-12-v2`
- **Retrieval**: LangChain's compression retriever with contextual reranking

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
# Each user gets isolated memory
user1_rag = NoBrainerRag(convo_id=f"user_{user1.id}")
user2_rag = NoBrainerRag(convo_id=f"user_{user2.id}")

# Their data never mixes
```

## FAQ

**Q: Do I need to keep the same NoBrainerRag object alive?**  
A: Nope! As long as you use the same `convo_id`, you can create new objects anytime and access the same data.

**Q: What happens if I use the same convo_id twice?**  
A: That's the point! Same ID = same memory. It's a feature, not a bug.

**Q: Can I use this in production?**  
A: Yeah, it's built on production-grade tools (Pinecone, LangChain, Ollama). Just make sure your Pinecone plan can handle your scale.

**Q: How much does Pinecone cost?**  
A: They have a generous free tier. Check [Pinecone pricing](https://www.pinecone.io/pricing/).

**Q: Can I change the embedding model?**  
A: Currently it uses `nomic-embed-text:v1.5`. Fork it if you need something different.

**Q: Is my data secure?**  
A: Data is stored in your Pinecone account. Use their security features + keep your API keys safe.

## Requirements

- Python 3.8+
- Pinecone API key
- Ollama installed locally
- `nomic-embed-text:v1.5` model pulled in Ollama

## Contributing

Found a bug? Have an idea? PRs welcome! Keep it simple though - the goal is "no brainer", not "all the features".

## License

MIT - do whatever you want with it.

## Support

If this saved you hours of work, star the repo ⭐ and help other devs find it!

---

Built with ❤️ for developers who just want things to work.

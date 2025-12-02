from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from typing import Optional
import os

load_dotenv()

pc = Pinecone()

class NoBrainerRag:
    def __init__(
            self,
            namespace,
            index_name: str,
            embedding_model = None,
            chunk_size: Optional[int] = 400,
            chunk_overlap: Optional[int] = 75,
            separators: Optional[list[str]] = ["\n\n", "\n", ".", ",", " ", ""],
            base_k: Optional[int] = 10,
            top_n: Optional[int] = 4,
            use_reranking: Optional[bool] = True,
            rerank_model: Optional[str] = "ms-marco-MiniLM-L-12-v2",
            pinecone_cloud: Optional[str] = "aws",
            pinecone_region: Optional[str] = "us-east-1",
            similarity_metric: Optional[str] = "cosine"
    ):
        """
        Initialize RAG for a conversation.
        
        Args:
            namespace: Unique namespace for this conversation
            index_name: Name of the Pinecone index to use
            embedding_model: Embedding model instance (default: OllamaEmbeddings with nomic-embed-text:v1.5)
            chunk_size: Size of text chunks (default: 400)
            chunk_overlap: Overlap between chunks (default: 75)
            separators: Text separators for chunking (default: ["\n\n","\n",".",","," ",""])
            base_k: Number of chunks to retrieve before reranking (default: 10)
            top_n: Number of chunks to return after reranking (default: 4)
            use_reranking: Enable FlashRank reranking (default: True)
            rerank_model: FlashRank model to use (default: "ms-marco-MiniLM-L-12-v2")
            pinecone_cloud: Pinecone cloud provider (default: "aws")
            pinecone_region: Pinecone region (default: "us-east-1")
            similarity_metric: Vector similarity metric (default: "cosine")
        """
        
        self.namespace = str(namespace)
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.base_k = base_k
        self.top_n = top_n
        self.use_reranking = use_reranking
        self.rerank_model = rerank_model
        
        # Set default embedding model if none provided
        self.embedding_model = embedding_model if embedding_model else OllamaEmbeddings(model="nomic-embed-text:v1.5")
        
        # Create index if it doesn't exist
        if not pc.has_index(self.index_name):
            pc.create_index(
                name=self.index_name,
                dimension=768,
                metric=similarity_metric,
                spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
            )
        
        self.index = pc.Index(self.index_name)
        
        self.vectorStore = PineconeVectorStore(
            index=self.index,
            embedding=self.embedding_model,
            namespace=str(namespace)
        )
    
    def add(self, text: str):
        """Insert text into vector database. Returns number of chunks created."""
        # Initialize splitter here
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        doc = [Document(page_content=text)]
        documents = splitter.split_documents(doc)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorStore.add_documents(documents=documents, ids=uuids)
        return f"Insertion Successful: {len(documents)} chunks created"
    
    def query(self, query: str):
        """
        Retrieve relevant documents for a query. Returns formatted string.
        Uses current instance variables for retrieval configuration.
        
        Args:
            query: Search query
        """
        # Build retriever fresh each time using current instance variables
        base_retriever = self.vectorStore.as_retriever(search_kwargs={"k": self.base_k})
        
        if self.use_reranking:
            retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=FlashrankRerank(model=self.rerank_model, top_n=self.top_n)
            )
        else:
            retriever = base_retriever
        
        doc_results = retriever.invoke(input=query)
        
        formatted_docs = []
        for i, doc in enumerate(doc_results, 1):
            formatted_docs.append(f"---DOCUMENT {i}---\n{doc.page_content}\n---END OF DOCUMENT {i}---")
        str_result = "\n\n".join(formatted_docs)
        return str_result
    
    def clear(self):
        """Delete all documents for this conversation."""
        self.index.delete(namespace=self.namespace, delete_all=True)
        return f"RAG memory of namespace '{self.namespace}' was successfully wiped out"
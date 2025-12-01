from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
from langchain_core.documents import Document
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from typing import Optional
import os

load_dotenv()

EmbeddingModel = OllamaEmbeddings(model="nomic-embed-text:v1.5")
pc = Pinecone()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not INDEX_NAME:
    raise ValueError("Missing PINECONE_API_KEY environment variable.")

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

class NoBrainerRag:
    def __init__(
            self,
            convo_id, 
            chunk_size: Optional[int]=400, 
            chunk_overlap: Optional[int] = 75,
            separators: Optional[list[str]] = ["\n\n","\n",".",","," ",""],
            base_k: Optional[int] = 10,
            top_n: Optional[int] = 4
                 ):
        
        """
        Initialize RAG for a conversation.
        
        Args:
            convo_id: Unique ID for this conversation
            chunk_size: Size of text chunks (default: 400)
            chunk_overlap: Overlap between chunks (default: 75)
            separators: Text separators for chunking (default: ["\n\n","\n",".",","," ",""])
            base_k: Number of chunks to retrieve before reranking (default: 10)
            top_n: Number of chunks to return after reranking (default: 4)
        """
        
        self.convo_id=str(convo_id)
        self.vectorStore = PineconeVectorStore(
            index=index,
            embedding=EmbeddingModel,
            namespace=str(convo_id)
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

        self.compression_retriever = ContextualCompressionRetriever(
            base_retriever=self.vectorStore.as_retriever(search_kwargs={"k": base_k}),
            base_compressor=FlashrankRerank(model="ms-marco-MiniLM-L-12-v2",top_n=top_n)
            )

    def insertIntoVectorDB(self,text: str):
        """Insert text into vector database. Returns number of chunks created."""
        doc = [Document(page_content=text)]
        documents = self.splitter.split_documents(doc)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorStore.add_documents(documents=documents,ids=uuids)
        return f"Insertion Successful {len(documents)} chunks created"
    
    def retrieveFromVectorDB(self,query:str):
        """Retrieve relevant documents for a query. Returns formatted string."""
        doc_results = self.compression_retriever.invoke(input=query)
        formatted_docs = []
        for i, doc in enumerate(doc_results, 1):
            formatted_docs.append(f"---DOCUMENT {i}---\n{doc.page_content}\n---END OF DOCUMENT {i}---")
        str_result = "\n\n".join(formatted_docs)
        return str_result

    def deleteConvoDB(self):
        """Delete all documents for this conversation."""
        index.delete(namespace=self.convo_id, delete_all=True)
        return f"Rag Memory of convo with{self.convo_id} id was successfully wiped out"

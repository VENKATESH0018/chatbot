import logging
from typing import List, Dict, Any, Optional
import os

from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import LLMClient
from utils import clean_text

logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system orchestrating document processing, vector storage, and LLM generation."""
    
    def __init__(self, document_processor: DocumentProcessor, vector_store: VectorStore, llm_client: LLMClient):
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for Paves Technologies context."""
        return """You are an AI assistant for Paves Technologies, a construction and infrastructure company. 
        You help employees and stakeholders find information from company documents including:
        - Project specifications and documentation
        - Company policies and procedures  
        - Technical guidelines and standards
        - Safety protocols and regulations
        - Contract information and legal documents
        
        Guidelines for responses:
        1. Always be professional and accurate
        2. If information is not available in the provided context, clearly state that
        3. Cite specific document sources when possible
        4. For safety-related questions, emphasize the importance of following proper protocols
        5. Use construction industry terminology appropriately
        6. Maintain confidentiality of sensitive business information
        
        Base your answers strictly on the provided context from Paves Technologies documents."""
    
    def add_document(self, file_path: str, filename: str) -> bool:
        """Add a document to the RAG system."""
        try:
            logger.info(f"Processing document: {filename}")
            
            # Extract text from document
            text_content = self.document_processor.extract_text(file_path)
            if not text_content:
                logger.error(f"No text extracted from {filename}")
                return False
            
            # Create chunks
            chunks = self.document_processor.create_chunks(text_content, filename)
            if not chunks:
                logger.error(f"No chunks created from {filename}")
                return False
            
            # Add chunks to vector store
            success = self.vector_store.add_documents(chunks)
            if success:
                logger.info(f"Successfully added {len(chunks)} chunks from {filename}")
                return True
            else:
                logger.error(f"Failed to add chunks from {filename} to vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error adding document {filename}: {str(e)}")
            return False
    
    def query(self, question: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """Query the RAG system with a question."""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search(question, top_k=top_k)
            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return None
            
            # Prepare context
            context = self._prepare_context(retrieved_docs)
            
            # Generate response using LLM
            response = self.llm_client.generate_response(
                question=question,
                context=context,
                system_prompt=self.system_prompt
            )
            
            if response:
                return {
                    "answer": response,
                    "sources": self._format_sources(retrieved_docs)
                }
            else:
                logger.error("Failed to generate response from LLM")
                return None
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return None
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            content = clean_text(doc.get("content", ""))
            metadata = doc.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            page = metadata.get("page", "N/A")
            
            context_part = f"[Document {i}: {filename}, Page {page}]\n{content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _format_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for display."""
        sources = []
        
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            sources.append({
                "filename": metadata.get("filename", "Unknown"),
                "page": metadata.get("page", "N/A"),
                "content": clean_text(doc.get("content", ""))[:300],
                "score": doc.get("score", 0.0)
            })
        
        return sources
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            vector_count = self.vector_store.get_collection_size()
            return {
                "total_chunks": vector_count,
                "vector_store_ready": vector_count > 0,
                "llm_ready": self.llm_client.is_ready()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "total_chunks": 0,
                "vector_store_ready": False,
                "llm_ready": False
            }

import logging
from typing import List, Dict, Any, Optional
import json
import os
import uuid
import math
from collections import Counter

logger = logging.getLogger(__name__)

class VectorStore:
    """Handle vector storage and similarity search using simple TF-IDF."""
    
    def __init__(self, collection_name: str = "paves_documents", embedding_model: str = "tfidf"):
        self.collection_name = collection_name
        self.storage_path = f"./simple_vector_db_{collection_name}.json"
        self.documents = []
        self.vocabulary = set()
        self.idf_scores = {}
        
        # Load existing data if available
        self._load_data()
        
        logger.info(f"Simple vector store initialized with {len(self.documents)} documents")
    
    def _load_data(self):
        """Load existing data from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.vocabulary = set(data.get('vocabulary', []))
                    self.idf_scores = data.get('idf_scores', {})
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
    
    def _save_data(self):
        """Save data to disk."""
        try:
            data = {
                'documents': self.documents,
                'vocabulary': list(self.vocabulary),
                'idf_scores': self.idf_scores
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [token for token in tokens if len(token) > 2]
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate term frequency."""
        token_count = Counter(tokens)
        total_tokens = len(tokens)
        return {token: count / total_tokens for token, count in token_count.items()}
    
    def _calculate_idf(self):
        """Calculate inverse document frequency for all terms."""
        if not self.documents:
            return
        
        total_docs = len(self.documents)
        term_doc_count = Counter()
        
        for doc in self.documents:
            doc_tokens = set(self._tokenize(doc['content']))
            for token in doc_tokens:
                term_doc_count[token] += 1
        
        self.idf_scores = {
            term: math.log(total_docs / count)
            for term, count in term_doc_count.items()
        }
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False
            
            added_count = 0
            for doc in documents:
                content = doc.get("content", "")
                if not content.strip():
                    continue
                
                # Add document with unique ID
                doc_entry = {
                    'id': str(uuid.uuid4()),
                    'content': content,
                    'metadata': doc.get("metadata", {}),
                    'tokens': self._tokenize(content)
                }
                
                self.documents.append(doc_entry)
                self.vocabulary.update(doc_entry['tokens'])
                added_count += 1
            
            if added_count > 0:
                # Recalculate IDF scores
                self._calculate_idf()
                self._save_data()
                logger.info(f"Successfully added {added_count} documents to vector store")
                return True
            else:
                logger.warning("No valid documents to add")
                return False
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def _calculate_cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two TF-IDF vectors."""
        # Get all unique terms
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        if not all_terms:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
        
        magnitude1 = math.sqrt(sum(val**2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _get_tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate TF-IDF vector for given tokens."""
        tf_scores = self._calculate_tf(tokens)
        tfidf_vector = {}
        
        for token in tokens:
            if token in self.idf_scores:
                tfidf_vector[token] = tf_scores[token] * self.idf_scores[token]
        
        return tfidf_vector
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search using TF-IDF and cosine similarity."""
        try:
            if not query.strip():
                logger.warning("Empty query provided")
                return []
            
            if not self.documents:
                logger.warning("No documents in the vector store")
                return []
            
            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                logger.warning("No valid tokens in query")
                return []
            
            # Calculate query TF-IDF vector
            query_tfidf = self._get_tfidf_vector(query_tokens)
            
            # Calculate similarities with all documents
            similarities = []
            for doc in self.documents:
                doc_tfidf = self._get_tfidf_vector(doc['tokens'])
                similarity = self._calculate_cosine_similarity(query_tfidf, doc_tfidf)
                
                if similarity > 0:  # Only include documents with some similarity
                    similarities.append({
                        "content": doc['content'],
                        "metadata": doc['metadata'],
                        "score": similarity
                    })
            
            # Sort by similarity score and return top_k
            similarities.sort(key=lambda x: x['score'], reverse=True)
            results = similarities[:top_k]
            
            logger.info(f"Retrieved {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def get_collection_size(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return len(self.documents)
        except Exception as e:
            logger.error(f"Error getting collection size: {str(e)}")
            return 0
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            if os.path.exists(self.storage_path):
                os.remove(self.storage_path)
            self.documents = []
            self.vocabulary = set()
            self.idf_scores = {}
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            self.documents = []
            self.vocabulary = set()
            self.idf_scores = {}
            self._save_data()
            logger.info(f"Cleared all documents from collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by metadata filters."""
        try:
            filtered_results = []
            
            for doc in self.documents:
                metadata = doc.get('metadata', {})
                match = True
                
                # Check if all filter criteria match
                for key, value in metadata_filter.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                
                if match:
                    filtered_results.append({
                        "content": doc['content'],
                        "metadata": metadata,
                        "score": 1.0  # No similarity score for metadata search
                    })
            
            # Limit results
            results = filtered_results[:top_k]
            logger.info(f"Found {len(results)} documents matching metadata filter")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            return []

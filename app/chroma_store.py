"""
Persistent ChromaDB client for vector storage
"""
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Any
from app.config import CHROMA_DIR


class ChromaStore:
    """Wrapper for ChromaDB operations"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
    
    def get_or_create_collection(self, collection_name: str):
        """Get or create a collection"""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(
        self,
        collection_name: str,
        doc_id: str,
        embedding: np.ndarray,
        text: str,
        metadata: Dict[str, Any]
    ):
        """Add a document to the collection"""
        collection = self.get_or_create_collection(collection_name)
        
        # Convert numpy array to list if needed
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
    
    def add_documents_batch(
        self,
        collection_name: str,
        doc_ids: List[str],
        embeddings: List[np.ndarray],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """Add multiple documents to the collection"""
        collection = self.get_or_create_collection(collection_name)
        
        # Convert numpy arrays to lists
        embeddings_list = [
            emb.tolist() if isinstance(emb, np.ndarray) else emb
            for emb in embeddings
        ]
        
        collection.add(
            ids=doc_ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        collection = self.get_or_create_collection(collection_name)
        
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def get_by_id(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        collection = self.get_or_create_collection(collection_name)
        
        try:
            results = collection.get(ids=[doc_id])
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'text': results['documents'][0],
                    'metadata': results['metadatas'][0],
                    'embedding': results['embeddings'][0] if 'embeddings' in results else None
                }
        except Exception:
            pass
        
        return None
    
    def delete_document(self, collection_name: str, doc_id: str):
        """Delete a document by ID"""
        collection = self.get_or_create_collection(collection_name)
        collection.delete(ids=[doc_id])
    
    def delete_collection(self, collection_name: str):
        """Delete an entire collection"""
        self.client.delete_collection(name=collection_name)
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def count_documents(self, collection_name: str) -> int:
        """Count documents in a collection"""
        collection = self.get_or_create_collection(collection_name)
        return collection.count()
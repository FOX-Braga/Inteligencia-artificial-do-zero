import numpy as np
from typing import List, Dict, Any

class SimpleVectorStore:
    """
    Armazenamento vetorial simples na memória (in-memory) 
    para demonstração usando NumPy (cosseno de similaridade).
    Pode ser substituído por ChromaDB, FAISS ou Pinecone.
    """
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]] = None):
        self.documents.extend(texts)
        self.embeddings.extend([np.array(e) for e in embeddings])
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in texts])
            
    def similarity_search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
            
        q_vec = np.array(query_embedding)
        
        # Calcular cosine similarity
        similarities = []
        for i, doc_vec in enumerate(self.embeddings):
            sim = np.dot(q_vec, doc_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(doc_vec) + 1e-9)
            similarities.append((sim, i))
            
        # Ordenar por maior similaridade
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for sim, idx in top_k:
            results.append({
                "score": sim,
                "text": self.documents[idx],
                "metadata": self.metadata[idx]
            })
            
        return results

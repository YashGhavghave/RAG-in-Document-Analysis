import os
import chromadb
from typing import List, Dict, Any, Optional
from rag.document_loader import DocumentChunk
from rag.embeddings import EmbeddingManager

class VectorStoreManager:
    """ChromaDB Vector Store Manager with persistent storage and metadata filtering."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "document_analysis_rag"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[DocumentChunk], embedding_manager: EmbeddingManager) -> int:
        """Indexes chunks into ChromaDB with embeddings and metadata."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        existing_count = self.collection.count()
        ids = [f"{c.metadata.get('chunk_id', 'chunk')}_{i}_{existing_count}" for i, c in enumerate(chunks)]

        embeddings = None
        if embedding_manager:
            embeddings = embedding_manager.embed_documents(texts)

        if embeddings:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        else:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        return len(chunks)

    def query(self, query_text: str, top_k: int = 4, embedding_manager: Optional[EmbeddingManager] = None, filter_source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs semantic similarity search."""
        if self.collection.count() == 0:
            return []

        where_filter = {"source": filter_source} if filter_source else None
        
        query_embedding = None
        if embedding_manager:
            query_embedding = embedding_manager.embed_query(query_text)

        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(top_k, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity_score": round(similarity * 100, 1)
                })

        return retrieved

    def get_all_sources(self) -> List[str]:
        """Returns unique list of document source names indexed."""
        count = self.collection.count()
        if count == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics of the vector store."""
        count = self.collection.count()
        sources = self.get_all_sources()
        return {
            "total_chunks": count,
            "document_count": len(sources),
            "documents": sources
        }

    def delete_source(self, source_name: str):
        """Deletes all chunks belonging to a specific document source."""
        self.collection.delete(where={"source": source_name})

    def clear(self):
        """Clears all indexed documents from vector store."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

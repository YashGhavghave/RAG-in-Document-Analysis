import os
from typing import List, Optional

class EmbeddingManager:
    """Handles vector embeddings using Google Gemini gemini-embedding-001 or Chroma fallback."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-embedding-001"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        # Normalize model name
        if "/" in model_name:
            self.model_name = model_name.split("/")[-1]
        else:
            self.model_name = model_name
        self.client = None
        self.legacy_active = False

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=self.api_key)
                    self.legacy_active = True
                except Exception as e:
                    print(f"Warning: Failed to configure Gemini Embeddings: {e}")

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Embed a list of document chunks."""
        if not texts:
            return []

        if self.client:
            try:
                from google.genai import types
                embeddings = []
                batch_size = 10
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    if hasattr(response, "embeddings") and response.embeddings:
                        for emb in response.embeddings:
                            embeddings.append(emb.values)
                    elif hasattr(response, "embedding"):
                        embeddings.append(response.embedding.values)
                return embeddings
            except Exception as e:
                print(f"Gemini client embedding error: {e}. Falling back to default.")

        if self.legacy_active:
            try:
                import google.generativeai as legacy_genai
                embeddings = []
                batch_size = 10
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    response = legacy_genai.embed_content(
                        model=f"models/{self.model_name}",
                        content=batch,
                        task_type="retrieval_document"
                    )
                    batch_embeddings = response.get("embedding", [])
                    embeddings.extend(batch_embeddings)
                return embeddings
            except Exception as e:
                print(f"Legacy Gemini embedding error: {e}")

        return None

    def embed_query(self, query: str) -> Optional[List[float]]:
        """Embed a search query."""
        if not query.strip():
            return None

        if self.client:
            try:
                from google.genai import types
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    return response.embeddings[0].values
                elif hasattr(response, "embedding"):
                    return response.embedding.values
            except Exception as e:
                print(f"Gemini query embedding error: {e}")

        if self.legacy_active:
            try:
                import google.generativeai as legacy_genai
                response = legacy_genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=query,
                    task_type="retrieval_query"
                )
                return response.get("embedding")
            except Exception as e:
                print(f"Legacy Gemini query embedding error: {e}")

        return None

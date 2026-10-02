import os
from typing import List, Dict, Any, Optional
import config

class RAGEngine:
    """Gemini RAG Engine with grounded citations, multi-level summarization, and document comparison."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = config.DEFAULT_CHAT_MODEL, temperature: float = config.DEFAULT_TEMPERATURE):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.temperature = temperature
        self.client = None
        self.legacy_model = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=self.api_key)
                    self.legacy_model = legacy_genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config=legacy_genai.types.GenerationConfig(
                            temperature=self.temperature
                        )
                    )
                except Exception as e:
                    print(f"Error initializing Gemini model: {e}")

    def is_configured(self) -> bool:
        return (self.client is not None or self.legacy_model is not None) and bool(self.api_key)

    def _generate(self, prompt: str) -> str:
        """Helper to invoke Gemini via new or legacy SDK."""
        if self.client:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=self.temperature)
            )
            return response.text
        elif self.legacy_model:
            response = self.legacy_model.generate_content(prompt)
            return response.text
        else:
            raise RuntimeError("Gemini is not configured")

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks with clear source and page citations for LLM context."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.get("metadata", {}).get("source", "Document")
            page = chunk.get("metadata", {}).get("page", 1)
            score = chunk.get("similarity_score", "N/A")
            text = chunk.get("text", "").strip()
            context_parts.append(
                f"--- [Snippet {i+1} | Source: {source} (Page/Section {page}) | Relevance: {score}%] ---\n{text}\n"
            )
        return "\n".join(context_parts)

    def answer_query(self, query: str, retrieved_chunks: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generates a grounded answer with citations using Gemini."""
        if not self.is_configured():
            return {
                "answer": "⚠️ Gemini API key is missing or not configured. Please enter your GEMINI_API_KEY in the sidebar.",
                "citations": []
            }

        if not retrieved_chunks:
            return {
                "answer": "No relevant document context found. Please ensure documents are uploaded and indexed.",
                "citations": []
            }

        context_str = self.format_context(retrieved_chunks)

        history_str = ""
        if chat_history:
            recent = chat_history[-4:]
            formatted_h = []
            for msg in recent:
                formatted_h.append(f"{msg['role'].capitalize()}: {msg['content']}")
            history_str = "\nConversation History:\n" + "\n".join(formatted_h) + "\n"

        prompt = f"""{config.RAG_SYSTEM_PROMPT}

{history_str}
Retrieved Document Context:
{context_str}

User Question: {query}

Please provide a comprehensive, strictly grounded answer with explicit document and page citations:
"""
        try:
            answer_text = self._generate(prompt)

            citations = []
            for chunk in retrieved_chunks:
                meta = chunk.get("metadata", {})
                citations.append({
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", 1),
                    "chunk_id": meta.get("chunk_id", ""),
                    "similarity": chunk.get("similarity_score", 0),
                    "snippet": chunk.get("text", "")[:250] + "..."
                })

            return {
                "answer": answer_text,
                "citations": citations,
                "raw_context": context_str
            }
        except Exception as e:
            return {
                "answer": f"❌ Error generating response from Gemini: {str(e)}",
                "citations": []
            }

    def generate_summary(self, chunks: List[Dict[str, Any]], doc_name: str) -> str:
        """Generates executive summary for a document or group of chunks."""
        if not self.is_configured():
            return "⚠️ Gemini API key is missing. Please enter your GEMINI_API_KEY in the sidebar."

        context_str = self.format_context(chunks)
        prompt = f"""You are an elite research analyst.
Document Name: {doc_name}

{config.SUMMARY_PROMPT}

Document Content:
{context_str}
"""
        try:
            return self._generate(prompt)
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"

    def compare_documents(self, doc_chunks_dict: Dict[str, List[Dict[str, Any]]]) -> str:
        """Compares multiple documents based on retrieved contexts."""
        if not self.is_configured():
            return "⚠️ Gemini API key is missing. Please enter your GEMINI_API_KEY in the sidebar."

        context_sections = []
        for doc_name, chunks in doc_chunks_dict.items():
            context_sections.append(f"=== DOCUMENT: {doc_name} ===\n{self.format_context(chunks)}")

        all_context = "\n\n".join(context_sections)
        prompt = f"""{config.COMPARISON_PROMPT}

{all_context}

Provide a structured, deep comparative synthesis:
"""
        try:
            return self._generate(prompt)
        except Exception as e:
            return f"❌ Error generating comparison: {str(e)}"

    def extract_entities_and_insights(self, chunks: List[Dict[str, Any]]) -> str:
        """Extracts structured entities, dates, metrics, and key terms."""
        if not self.is_configured():
            return "⚠️ Gemini API key is missing. Please enter your GEMINI_API_KEY in the sidebar."

        context_str = self.format_context(chunks)
        prompt = f"""{config.ENTITY_EXTRACTION_PROMPT}

Document Context:
{context_str}
"""
        try:
            return self._generate(prompt)
        except Exception as e:
            return f" Error extracting entities: {str(e)}"

    def generate_quiz(self, chunks: List[Dict[str, Any]]) -> str:
        """Generates an interactive quiz and comprehension check."""
        if not self.is_configured():
            return " Gemini API key is missing. Please enter your GEMINI_API_KEY in the sidebar."

        context_str = self.format_context(chunks)
        prompt = f"""{config.QUIZ_PROMPT}

Document Context:
{context_str}
"""
        try:
            return self._generate(prompt)
        except Exception as e:
            return f" Error generating quiz: {str(e)}"

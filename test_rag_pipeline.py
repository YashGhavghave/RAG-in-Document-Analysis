import os
import unittest
import tempfile
import pandas as pd

from rag.document_loader import DocumentLoader, DocumentChunk
from rag.text_splitter import RecursiveCharacterSplitter
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.rag_engine import RAGEngine

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_text_path = "samples/sample_report.txt"
        self.ai_text_path = "samples/ai_overview.txt"
        self.temp_db_dir = tempfile.mkdtemp()
        self.vector_store = VectorStoreManager(persist_directory=self.temp_db_dir, collection_name="test_collection")
        self.embedding_manager = EmbeddingManager(api_key="")

    def test_document_loader_txt(self):
        chunks = DocumentLoader.load_file(self.sample_text_path)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0].metadata["file_type"], "txt")
        self.assertEqual(chunks[0].metadata["page"], 1)

    def test_document_loader_csv(self):
        # Create a temp CSV
        temp_csv = os.path.join(self.temp_db_dir, "test_data.csv")
        df = pd.DataFrame({
            "Metric": ["Revenue", "Growth", "Customers"],
            "Value": [1000000, 25.5, 5400]
        })
        df.to_csv(temp_csv, index=False)
        
        chunks = DocumentLoader.load_file(temp_csv)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0].metadata["file_type"], "csv")
        self.assertEqual(chunks[0].metadata["row_count"], 3)

    def test_recursive_splitter(self):
        raw_chunks = DocumentLoader.load_file(self.sample_text_path)
        splitter = RecursiveCharacterSplitter(chunk_size=300, chunk_overlap=50)
        split_chunks = splitter.split_documents(raw_chunks)
        
        self.assertTrue(len(split_chunks) >= len(raw_chunks))
        for chunk in split_chunks:
            self.assertIn("chunk_id", chunk.metadata)
            self.assertIn("source", chunk.metadata)
            self.assertTrue(len(chunk.text) > 0)

    def test_vector_store_indexing_and_query(self):
        # Load both sample documents
        raw_1 = DocumentLoader.load_file(self.sample_text_path)
        raw_2 = DocumentLoader.load_file(self.ai_text_path)
        
        splitter = RecursiveCharacterSplitter(chunk_size=400, chunk_overlap=80)
        chunks = splitter.split_documents(raw_1 + raw_2)
        
        # Add to vector store
        added_count = self.vector_store.add_documents(chunks, self.embedding_manager)
        self.assertEqual(added_count, len(chunks))
        
        # Verify stats
        stats = self.vector_store.get_stats()
        self.assertEqual(stats["total_chunks"], len(chunks))
        self.assertEqual(stats["document_count"], 2)
        
        # Query
        results = self.vector_store.query("solar power clean energy investment", top_k=3, embedding_manager=self.embedding_manager)
        self.assertTrue(len(results) > 0)
        self.assertTrue("similarity_score" in results[0])
        self.assertTrue("metadata" in results[0])

    def test_rag_engine_citation_formatting(self):
        engine = RAGEngine(api_key="")
        mock_chunks = [
            {
                "text": "Solar installations increased 42% in 2025.",
                "metadata": {"source": "energy_report.pdf", "page": 3, "chunk_id": "chunk_1"},
                "similarity_score": 92.5
            }
        ]
        context = engine.format_context(mock_chunks)
        self.assertIn("energy_report.pdf", context)
        self.assertIn("Page/Section 3", context)
        self.assertIn("92.5%", context)

if __name__ == "__main__":
    unittest.main()

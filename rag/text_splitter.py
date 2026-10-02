import re
from typing import List
from rag.document_loader import DocumentChunk

class RecursiveCharacterSplitter:
    """Recursively splits text into chunks using hierarchical separators."""
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits = []
        for s in splits:
            if s:
                if len(s) < self.chunk_size:
                    good_splits.append(s)
                else:
                    if new_separators:
                        other_info = self._split_text(s, new_separators)
                        good_splits.extend(other_info)
                    else:
                        good_splits.append(s)

        merged = []
        current_chunk = []
        current_len = 0

        for s in good_splits:
            add_len = len(s) + (len(separator) if current_chunk else 0)
            if current_len + add_len > self.chunk_size and current_chunk:
                doc = separator.join(current_chunk).strip()
                if doc:
                    merged.append(doc)
                
                while current_chunk and current_len > self.chunk_overlap:
                    popped = current_chunk.pop(0)
                    current_len -= len(popped) + len(separator)

            current_chunk.append(s)
            current_len += len(s) + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            doc = separator.join(current_chunk).strip()
            if doc:
                merged.append(doc)

        return merged

    def split_documents(self, documents: List[DocumentChunk]) -> List[DocumentChunk]:
        split_docs = []
        chunk_counter = 0
        for doc in documents:
            text = doc.text.strip()
            if not text:
                continue
            
            raw_chunks = self._split_text(text, self.separators)
            for idx, chunk_text in enumerate(raw_chunks):
                if not chunk_text.strip():
                    continue
                chunk_counter += 1
                meta = dict(doc.metadata)
                meta["chunk_id"] = f"{meta.get('source', 'doc')}_p{meta.get('page', 1)}_c{idx+1}"
                meta["chunk_index"] = idx + 1
                meta["global_chunk_id"] = chunk_counter
                meta["char_length"] = len(chunk_text)
                split_docs.append(DocumentChunk(text=chunk_text, metadata=meta))
        return split_docs

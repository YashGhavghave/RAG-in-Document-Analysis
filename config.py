import os
from dotenv import load_dotenv

load_dotenv()

# Gemini Models
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
AVAILABLE_CHAT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# RAG & Chunking Defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.2

# System Prompt Templates
RAG_SYSTEM_PROMPT = """You are an expert AI Document Analyst.
Your task is to answer user queries accurately based ONLY on the provided context retrieved from documents.

Guidelines:
1. Ground your answers strictly in the provided Context.
2. Cite the exact Source Document and Page/Section number where the information was found (e.g. [Source: report.pdf, Page 2]).
3. If the context does not contain enough information to answer the question, state clearly: "Based on the provided documents, I could not find information regarding this." Do not fabricate facts.
4. Structure your response with clear headings, bullet points, and concise summaries where applicable.
5. If there are contradictions or differing perspectives across documents, highlight them clearly.
"""

SUMMARY_PROMPT = """Analyze the provided document context and generate a structured executive summary with:
1. 📌 **Executive Overview**: High-level synthesis of what the document covers.
2. 🔑 **Core Themes & Key Insights**: Bullet points highlighting the most important findings or concepts.
3. 📊 **Key Metrics, Dates & Figures**: Any quantitative data, milestones, or key metrics mentioned.
4. ⚠️ **Risks & Challenges**: Identified limitations, risks, or warning points.
5. 💡 **Actionable Takeaways & Next Steps**: Practical recommendations based on the content.
"""

COMPARISON_PROMPT = """Analyze and compare the following documents based on the provided retrieved context.
Structure your comparison with:
1. 🎯 **Common Ground & Overlaps**: Shared themes, conclusions, or facts.
2. ⚡ **Key Differences & Contrasts**: Direct contrast between the documents on key topics.
3. 📊 **Comparative Matrix / Key Metrics**: Compare metrics, timelines, or arguments side-by-side.
4. 🏁 **Conclusion & Summary Verdict**: Overall synthesis of how these documents complement or contradict each other.
"""

ENTITY_EXTRACTION_PROMPT = """Analyze the document context and extract key structured information in markdown format:
- **Organizations & Companies**: List with brief role/context
- **Key People & Stakeholders**: List with titles or actions
- **Dates, Deadlines & Timelines**: Chronological list of key dates
- **Financial & Numerical Data**: Important numbers, currency amounts, statistics
- **Key Terms & Definitions**: Domain-specific glossary items
"""

QUIZ_PROMPT = """Based on the provided document context, generate 5 challenging quiz questions with detailed explanations and answers to test comprehension.
Format each question as:
### Question [N]: [Question Title]
- **Options**: A, B, C, D
- **Correct Answer**: [Letter]
- **Explanation**: [Why this answer is correct based on the text]
- **Source Citation**: [Document and page/section reference]
"""

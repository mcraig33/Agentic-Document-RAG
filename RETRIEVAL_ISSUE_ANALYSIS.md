# Retrieval Issue Analysis: Why Company Information Isn't Being Found

## Problem Statement

When asking questions like "What companies are here?" or "What 10-ks are included?", the LLM responds that the information isn't in the context, even though the documents (10-K filings) clearly contain company identification information.

## Root Cause Analysis

### 1. **Semantic Search Mismatch**
The question "What companies are here?" is asking a **meta-question** about the document collection, but semantic search is looking for chunks that are semantically similar to this question. Chunks containing "Apple Inc." or "Rivian" are not semantically similar to the phrase "What companies are here?" - they're about the companies themselves, not about identifying companies in a collection.

**Example:**
- Question: "What companies are here?"
- Chunk: "Apple Inc. (Exact name of Registrant as specified in its charter)"
- Semantic similarity: **LOW** - these are about different things

### 2. **Limited Retrieval (k=5)**
With only 5 chunks retrieved, there's a high chance that none of them contain company identification information, especially if:
- Company names appear early in documents (cover pages)
- The question doesn't semantically match those early chunks
- Most retrieved chunks are from later sections

### 3. **Chunk Fragmentation**
10-K documents are parsed into many small chunks. Company identification might be split across multiple chunks or appear in chunks that don't rank highly for the query.

## Solutions Implemented

### ✅ Solution 1: Increased Retrieval Count
- **Changed**: `k=5` → `k=10`
- **Why**: More chunks = higher probability of retrieving chunks with company information
- **Trade-off**: Slightly more tokens, but better coverage

### ✅ Solution 2: Improved Prompt
- **Removed**: Filename-based company identification
- **Added**: Explicit instructions to look for:
  - Company names, legal entity names, registrant names
  - Trading symbols (AAPL, RIVN, etc.)
  - Phrases like "FORM 10-K", "Annual Report"
  - Information in cover pages and business descriptions
- **Why**: Guides the LLM to extract information that IS in the context

### ✅ Solution 3: Removed Source Metadata from Context
- **Changed**: Removed `[Source: filename, Page X]` prefixes
- **Why**: Focus LLM on document content, not filenames

## Additional Solutions to Consider

### 🔄 Solution 4: Hybrid Retrieval (Recommended Next Step)
Combine semantic search with keyword search for meta-questions:

```python
# For questions about document collection, also search for:
# - Company name patterns: "Inc.", "Corporation", "Company"
# - Document type patterns: "FORM 10-K", "Annual Report"
# - Trading symbols: Extract from context or metadata
```

### 🔄 Solution 5: Query Rewriting
Transform meta-questions into content-focused queries:
- "What companies are here?" → "What is the name of the registrant company?"
- "What 10-ks are included?" → "What type of SEC filing is this document?"

### 🔄 Solution 6: Metadata Filtering
Use ChromaDB metadata to directly answer collection-level questions:
- Query metadata for unique `source_file` values
- Extract company names from filenames as fallback
- Use this for meta-questions, content for content questions

### 🔄 Solution 7: Multi-Stage Retrieval
1. First retrieve chunks semantically similar to the question
2. If question is meta-level, also retrieve:
   - First page chunks (cover page often has company info)
   - Chunks containing company indicators ("Inc.", "Corporation")
   - Chunks with high "registrant" or "company name" scores

## Testing Recommendations

1. **Test with explicit company questions:**
   - "What is the name of the company in this 10-K?"
   - "What company filed this document?"

2. **Test retrieval quality:**
   - Run `check_chromadb.py` to see what chunks are retrieved
   - Verify company identification chunks are in the database
   - Check if they're being retrieved for meta-questions

3. **Compare retrieval strategies:**
   - Current: Pure semantic search (k=10)
   - Proposed: Hybrid semantic + keyword search
   - Proposed: Query rewriting for meta-questions

## Expected Improvement

With the current changes (k=10, improved prompt):
- **Better**: More chunks retrieved increases chance of finding company info
- **Better**: Improved prompt helps LLM extract info that IS in context
- **Still needs work**: Meta-questions may still struggle with semantic search

**Next step**: Implement hybrid retrieval or query rewriting for best results.

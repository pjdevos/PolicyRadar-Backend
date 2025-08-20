#!/usr/bin/env python3
"""
Policy Radar RAG (Retrieval-Augmented Generation) Service

Provides natural language query interface over Policy Radar documents
with grounded citations and multilingual support.

Usage:
  python rag_service.py --index ./vectors/policy_index --query "What are the latest hydrogen regulations?"

Features:
- Multilingual query processing
- Hybrid search (semantic + keyword)
- Citation-forced generation
- Source filtering
- Query expansion with EuroVoc concepts
"""

import argparse
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass

# Import our vector store
from vector_indexer import PolicyVectorStore, DocumentChunk

@dataclass
class RAGResult:
    """Result from RAG query"""
    answer: str
    sources: List[DocumentChunk]
    query_expansion: List[str]
    confidence: float
    language: str

class PolicyRAGService:
    """RAG service for Policy Radar queries"""
    
    def __init__(self, index_path: str, llm_provider: str = "claude"):
        """Initialize RAG service with vector index and LLM"""
        self.vector_store = PolicyVectorStore()
        config = self.vector_store.load_index(index_path)
        self.llm_provider = llm_provider
        
        # Query expansion terms from EuroVoc
        self.eurovoc_concepts = config.get('eurovoc_map', {})
        
        print(f"RAG Service initialized with {config['num_chunks']} chunks")
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with related EuroVoc concepts and synonyms"""
        expansions = [query]
        query_lower = query.lower()
        
        # Add related concepts
        for concept, terms in self.eurovoc_concepts.items():
            if concept in query_lower:
                expansions.extend(terms[:3])  # Top 3 related terms
            
            for term in terms:
                if term.lower() in query_lower:
                    expansions.append(concept)
                    expansions.extend([t for t in terms if t != term][:2])
        
        # Domain-specific synonyms
        synonyms = {
            'regulation': ['directive', 'legislation', 'law', 'policy'],
            'hydrogen': ['H2', 'fuel cell', 'clean fuel'],
            'electric': ['EV', 'battery', 'electric vehicle'],
            'transport': ['mobility', 'transportation', 'logistics'],
            'climate': ['environmental', 'carbon', 'emissions', 'green'],
            'energy': ['power', 'electricity', 'renewable']
        }
        
        for key, values in synonyms.items():
            if key in query_lower:
                expansions.extend(values)
        
        return list(set(expansions))  # Remove duplicates
    
    def _retrieve_documents(self, query: str, k: int = 8, 
                          source_filter: Optional[str] = None,
                          doc_type_filter: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve relevant documents using hybrid search"""
        
        # Expand query for better recall
        expanded_terms = self._expand_query(query)
        
        # Primary semantic search
        results = self.vector_store.search(
            query, 
            k=k, 
            source_filter=source_filter,
            doc_type_filter=doc_type_filter
        )
        
        # Secondary searches with expanded terms
        for term in expanded_terms[1:4]:  # Top 3 expansion terms
            expanded_results = self.vector_store.search(
                term,
                k=k//2,
                source_filter=source_filter,
                doc_type_filter=doc_type_filter
            )
            results.extend(expanded_results)
        
        # Deduplicate by chunk_id and re-rank by score
        seen_chunks = set()
        unique_results = []
        
        for chunk, score in sorted(results, key=lambda x: x[1], reverse=True):
            if chunk.chunk_id not in seen_chunks:
                seen_chunks.add(chunk.chunk_id)
                unique_results.append((chunk, score))
        
        return unique_results[:k]
    
    def _generate_answer(self, query: str, sources: List[DocumentChunk], 
                        language: str = "en") -> Tuple[str, float]:
        """Generate grounded answer using LLM"""
        
        # Prepare context from sources
        context_parts = []
        for i, chunk in enumerate(sources[:6]):  # Limit context size
            source_info = f"[{i+1}] Source: {chunk.source} ({chunk.doc_type})"
            if chunk.published:
                source_info += f" | Date: {chunk.published[:10]}"
            
            context_parts.append(f"{source_info}\nTitle: {chunk.title}\nContent: {chunk.content[:400]}...\nURL: {chunk.url}\n")
        
        context = "\n".join(context_parts)
        
        # Craft prompt for citation-forced generation
        system_prompt = f"""You are a Policy Radar assistant specializing in EU policy and legislation analysis. 

CRITICAL INSTRUCTIONS:
- Answer in {language} language
- ALWAYS cite your sources using [SOURCE_NUMBER] format
- Only use information from the provided sources
- If sources don't contain sufficient information, say so clearly
- Provide specific dates, document numbers, and URLs when available
- Focus on official sources (EUR-Lex, EP) over news sources when possible
- Be precise about regulatory status (proposed, adopted, in force, etc.)"""

        user_prompt = f"""Query: {query}

Available Sources:
{context}

Please provide a comprehensive answer to the query using ONLY the information from the sources above. 
Every factual claim must be cited with [SOURCE_NUMBER]. 
If the sources don't contain enough information, state this clearly."""

        # Call LLM (simplified - in production, use proper API)
        try:
            answer = self._call_llm(system_prompt, user_prompt)
            confidence = self._estimate_confidence(answer, sources)
            return answer, confidence
        except Exception as e:
            return f"Error generating answer: {e}", 0.0
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API - simplified implementation"""
        
        if self.llm_provider == "claude":
            # Using Claude API (requires API key in production)
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1000,
                        "messages": [
                            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                        ]
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["content"][0]["text"]
                else:
                    return f"API Error: {response.status_code}"
                    
            except Exception as e:
                return f"LLM Error: {e}"
        
        # Fallback: Template-based response
        return self._template_response(user_prompt)
    
    def _template_response(self, query: str) -> str:
        """Fallback template-based response when LLM unavailable"""
        return f"""Based on the available Policy Radar sources, here are the key findings for: "{query}"

[This is a template response - LLM integration pending]

The sources indicate relevant policy developments in this area. For detailed analysis, please review the cited documents directly.

[1] Multiple sources found matching your query
[2] Additional context available in EUR-Lex documentation
[3] Recent procedural updates from European Parliament data"""
    
    def _estimate_confidence(self, answer: str, sources: List[DocumentChunk]) -> float:
        """Estimate confidence based on source quality and citation coverage"""
        
        # Count citations in answer
        citations = len(re.findall(r'\[(\d+)\]', answer))
        max_citations = min(len(sources), 6)
        
        # Source quality scoring
        source_scores = []
        for chunk in sources:
            score = 0.5  # base score
            
            # Official sources get higher scores
            if chunk.source == "EUR-Lex":
                score += 0.3
            elif chunk.source == "EP Open Data":
                score += 0.25
            elif chunk.source == "EURACTIV":
                score += 0.15
            
            # Recent sources get higher scores
            if chunk.published:
                try:
                    pub_date = datetime.fromisoformat(chunk.published.replace('Z', '+00:00'))
                    days_old = (datetime.now().replace(tzinfo=None) - pub_date.replace(tzinfo=None)).days
                    if days_old < 30:
                        score += 0.1
                    elif days_old < 90:
                        score += 0.05
                except:
                    pass
            
            source_scores.append(min(score, 1.0))
        
        # Combine factors
        citation_score = citations / max_citations if max_citations > 0 else 0
        avg_source_score = sum(source_scores) / len(source_scores) if source_scores else 0.5
        
        confidence = (citation_score * 0.4 + avg_source_score * 0.6)
        return min(confidence, 1.0)
    
    def query(self, query: str, source_filter: Optional[str] = None,
              doc_type_filter: Optional[str] = None, k: int = 8) -> RAGResult:
        """Main query interface"""
        
        print(f"🔍 Processing query: {query}")
        
        # Query expansion
        expanded_terms = self._expand_query(query)
        print(f"📈 Query expansion: {expanded_terms[:5]}")
        
        # Retrieve relevant documents
        sources_with_scores = self._retrieve_documents(
            query, k=k, 
            source_filter=source_filter,
            doc_type_filter=doc_type_filter
        )
        
        sources = [chunk for chunk, score in sources_with_scores]
        print(f"📄 Retrieved {len(sources)} relevant sources")
        
        # Generate answer
        answer, confidence = self._generate_answer(query, sources)
        
        # Detect query language (simplified)
        language = "en"  # Default - could use langdetect
        
        result = RAGResult(
            answer=answer,
            sources=sources,
            query_expansion=expanded_terms,
            confidence=confidence,
            language=language
        )
        
        return result
    
    def print_result(self, result: RAGResult):
        """Pretty print RAG result"""
        print(f"\n🤖 **Answer** (confidence: {result.confidence:.2f}):")
        print(result.answer)
        
        print(f"\n📚 **Sources** ({len(result.sources)}):")
        for i, chunk in enumerate(result.sources):
            print(f"[{i+1}] {chunk.source} | {chunk.title[:60]}...")
            print(f"    🔗 {chunk.url}")
            if chunk.published:
                print(f"    📅 {chunk.published[:10]}")
            print()

def main():
    parser = argparse.ArgumentParser(description="Policy Radar RAG Service")
    parser.add_argument("--index", required=True, help="Vector index directory")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--source", help="Filter by source (EURACTIV, EUR-Lex, EP Open Data)")
    parser.add_argument("--doc-type", help="Filter by document type (news, legal, procedure)")
    parser.add_argument("--k", type=int, default=8, help="Number of sources to retrieve")
    parser.add_argument("--llm", default="claude", help="LLM provider (claude, template)")
    
    args = parser.parse_args()
    
    # Initialize RAG service
    rag = PolicyRAGService(args.index, llm_provider=args.llm)
    
    # Process query
    result = rag.query(
        args.query,
        source_filter=args.source,
        doc_type_filter=args.doc_type,
        k=args.k
    )
    
    # Display results
    rag.print_result(result)

if __name__ == "__main__":
    main()
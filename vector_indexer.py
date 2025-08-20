#!/usr/bin/env python3
"""
Policy Radar Vector Indexing Module

Reads JSONL output from poc_policy_radar.py and creates vector embeddings
for semantic search and RAG capabilities.

Usage:
  python vector_indexer.py --input ./out/items.jsonl --index ./vectors/policy_index

Features:
- Multilingual embeddings (sentence-transformers)
- Chunking for large documents
- EuroVoc concept integration
- Metadata preservation for filtering
- FAISS/Qdrant support
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# Core ML libraries
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Text processing
import re
from langdetect import detect
import requests

@dataclass
class DocumentChunk:
    """Represents a searchable chunk of a document"""
    chunk_id: str           # doc_id + chunk_index
    doc_id: str            # original DocItem.id
    source: str            # EURACTIV/EUR-Lex/EP
    doc_type: str          # news/legal/procedure
    title: str
    content: str           # the actual text to embed
    language: str
    url: str
    published: Optional[str]
    topics: List[str]
    eurovoc_concepts: List[str]  # NEW: EuroVoc classifications
    chunk_index: int       # position in document
    metadata: Dict[str, Any]     # preserved from DocItem.extra

class PolicyVectorStore:
    """Vector store for Policy Radar documents"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """Initialize with multilingual sentence transformer"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # FAISS index for similarity search
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Metadata store (indexed by vector position)
        self.chunks: List[DocumentChunk] = []
        self.eurovoc_map = self._load_eurovoc_concepts()
        
    def _load_eurovoc_concepts(self) -> Dict[str, List[str]]:
        """Load EuroVoc concept mappings for topic enrichment"""
        # Simplified concept map - in production, load from official EuroVoc SKOS
        concepts = {
            'energy': ['renewable energy', 'energy policy', 'energy security', 'energy transition'],
            'transport': ['sustainable transport', 'electric vehicles', 'public transport', 'mobility'],
            'hydrogen': ['hydrogen economy', 'fuel cells', 'clean energy', 'energy storage'],
            'climate': ['climate change', 'greenhouse gases', 'carbon neutrality', 'emissions'],
            'environment': ['environmental policy', 'pollution control', 'biodiversity', 'circular economy']
        }
        return concepts
    
    def _detect_eurovoc_concepts(self, text: str, topics: List[str]) -> List[str]:
        """Extract relevant EuroVoc concepts from text and existing topics"""
        concepts = set()
        text_lower = text.lower()
        
        # Add concepts based on explicit topics
        for topic in topics:
            topic_lower = topic.lower()
            if topic_lower in self.eurovoc_map:
                concepts.update(self.eurovoc_map[topic_lower])
        
        # Keyword-based concept detection
        for concept_group, keywords in self.eurovoc_map.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    concepts.add(concept_group)
                    concepts.update(keywords[:2])  # Add top 2 specific concepts
        
        return list(concepts)
    
    def _chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into semantically meaningful chunks"""
        if not text or len(text) <= max_length:
            return [text] if text else []
        
        # Try to split on sentences first
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Fallback: character-based chunking for very long sentences
        if not chunks and text:
            for i in range(0, len(text), max_length):
                chunks.append(text[i:i+max_length])
        
        return chunks
    
    def add_documents(self, jsonl_path: str) -> int:
        """Load documents from PoC output and add to vector store"""
        print(f"Loading documents from {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            documents = [json.loads(line) for line in f if line.strip()]
        
        print(f"Processing {len(documents)} documents...")
        all_texts = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            # Create searchable text from title + summary + body
            searchable_parts = [
                doc.get('title', ''),
                doc.get('summary', ''),
                doc.get('body_text', '')
            ]
            searchable_text = ' '.join(filter(None, searchable_parts))
            
            if not searchable_text.strip():
                continue
            
            # Detect language if not provided
            try:
                language = doc.get('language') or detect(searchable_text)
            except:
                language = 'en'  # fallback
            
            # Extract EuroVoc concepts
            eurovoc_concepts = self._detect_eurovoc_concepts(
                searchable_text, 
                doc.get('topics', [])
            )
            
            # Chunk the document
            chunks = self._chunk_text(searchable_text, max_length=400)
            
            for i, chunk_text in enumerate(chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{doc['id']}_{i}",
                    doc_id=doc['id'],
                    source=doc['source'],
                    doc_type=doc['doc_type'],
                    title=doc['title'],
                    content=chunk_text,
                    language=language,
                    url=doc['url'],
                    published=doc.get('published'),
                    topics=doc.get('topics', []),
                    eurovoc_concepts=eurovoc_concepts,
                    chunk_index=i,
                    metadata=doc.get('extra', {})
                )
                
                self.chunks.append(chunk)
                all_texts.append(chunk_text)
        
        # Generate embeddings in batches
        print("Generating embeddings...")
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding batches"):
            batch = all_texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
        
        # Add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        print(f"Indexed {len(embeddings)} chunks from {len(documents)} documents")
        return len(embeddings)
    
    def search(self, query: str, k: int = 10, source_filter: Optional[str] = None, 
               doc_type_filter: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant document chunks"""
        
        # Embed the query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k * 3)  # Get more for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            chunk = self.chunks[idx]
            
            # Apply filters
            if source_filter and chunk.source != source_filter:
                continue
            if doc_type_filter and chunk.doc_type != doc_type_filter:
                continue
            
            results.append((chunk, float(score)))
            
            if len(results) >= k:
                break
        
        return results
    
    def save_index(self, index_path: str):
        """Save the vector index and metadata"""
        os.makedirs(index_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(index_path, "faiss.index"))
        
        # Save chunks metadata
        with open(os.path.join(index_path, "chunks.pkl"), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save configuration
        config = {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'unknown',
            'dimension': self.dimension,
            'num_chunks': len(self.chunks),
            'eurovoc_map': self.eurovoc_map
        }
        
        with open(os.path.join(index_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load a previously saved index"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(index_path, "faiss.index"))
        
        # Load chunks metadata
        with open(os.path.join(index_path, "chunks.pkl"), 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load configuration
        with open(os.path.join(index_path, "config.json"), 'r') as f:
            config = json.load(f)
        
        print(f"Loaded index from {index_path}: {config['num_chunks']} chunks")
        return config

def main():
    parser = argparse.ArgumentParser(description="Policy Radar Vector Indexer")
    parser.add_argument("--input", required=True, help="Input JSONL file from PoC")
    parser.add_argument("--index", required=True, help="Output index directory")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                       help="Sentence transformer model")
    
    args = parser.parse_args()
    
    # Initialize vector store
    store = PolicyVectorStore(model_name=args.model)
    
    # Add documents and create index
    num_chunks = store.add_documents(args.input)
    
    # Save the index
    store.save_index(args.index)
    
    print(f"\n✅ Vector indexing complete!")
    print(f"   📊 Processed chunks: {num_chunks}")
    print(f"   💾 Index saved to: {args.index}")
    print(f"   🔍 Ready for semantic search!")
    
    # Demo search
    print(f"\n🔍 Demo search...")
    results = store.search("hydrogen fuel cells", k=3)
    for i, (chunk, score) in enumerate(results):
        print(f"{i+1}. [{chunk.source}] {chunk.title[:50]}... (score: {score:.3f})")

if __name__ == "__main__":
    main()
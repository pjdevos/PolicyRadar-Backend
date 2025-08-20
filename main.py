#!/usr/bin/env python3
"""
Policy Radar Backend - Enhanced Version

Enhanced FastAPI backend with RAG capabilities, vector search, and comprehensive configuration.
Maintains compatibility with existing endpoints while adding advanced features.

Usage:
  python main.py
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
from functools import wraps
import time
from collections import defaultdict
from threading import Lock

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Try to import advanced configuration system
try:
    from config.settings import settings, validate_startup_config
    ADVANCED_CONFIG = True
    print("[OK] Advanced configuration system loaded")
except ImportError:
    print("[WARN] Advanced configuration not found, using fallback config")
    ADVANCED_CONFIG = False
    # Fallback configuration
    class FallbackSettings:
        class Database:
            DATA_DIR = Path("./data")
            VECTOR_DB_PATH = Path("./vectors")
            VECTOR_INDEX_NAME = "policy_index"
        
        class API:
            HOST = "0.0.0.0"
            PORT = int(os.getenv("PORT", 8000))
            CORS_ORIGINS = ["http://localhost:3000", "https://*.vercel.app", "https://*.railway.app"]
            DEFAULT_RATE_LIMIT = 100
            RAG_RATE_LIMIT = 10
            INGEST_RATE_LIMIT = 5
        
        class Security:
            MAX_QUERY_LENGTH = 500
            MAX_TOPIC_LENGTH = 50
            TRUSTED_HOSTS = ["localhost", "127.0.0.1", "*.up.railway.app", "*.vercel.app"]
        
        def is_development(self):
            return os.getenv("ENVIRONMENT", "development") != "production"
        
        def is_production(self):
            return os.getenv("ENVIRONMENT", "development") == "production"
    
    settings = FallbackSettings()
    settings.database = settings.Database()
    settings.api = settings.API()
    settings.security = settings.Security()

# Try to import advanced RAG and vector services
try:
    from vector_indexer import PolicyVectorStore
    from rag_service import PolicyRAGService
    RAG_AVAILABLE = True
    print("[OK] RAG and vector search services loaded")
except ImportError:
    print("[WARN] RAG services not found, using mock implementations")
    RAG_AVAILABLE = False
    # Mock classes for compatibility
    class PolicyVectorStore:
        def __init__(self):
            pass
        def load_index(self, path):
            pass
    
    class PolicyRAGService:
        def __init__(self, path):
            pass
        def query(self, query, **kwargs):
            # Mock RAG response
            class MockResult:
                def __init__(self):
                    self.answer = f"Mock response for: {query}"
                    self.sources = []
                    self.query_expansion = [query]
                    self.confidence = 0.8
            return MockResult()

# Configuration paths
if ADVANCED_CONFIG:
    DATA_DIR = settings.database.DATA_DIR
    VECTORS_DIR = settings.database.VECTOR_DB_PATH
    INDEX_PATH = VECTORS_DIR / settings.database.VECTOR_INDEX_NAME
else:
    DATA_DIR = Path("./data")
    VECTORS_DIR = Path("./vectors") 
    INDEX_PATH = VECTORS_DIR / "policy_index"

# Rate limiting setup
if ADVANCED_CONFIG:
    rate_limit_storage = defaultdict(lambda: {"count": 0, "reset_time": 0})
    rate_limit_lock = Lock()
    
    RATE_LIMITS = {
        "default": {"requests": settings.api.DEFAULT_RATE_LIMIT, "window": 3600},
        "/api/rag/query": {"requests": settings.api.RAG_RATE_LIMIT, "window": 3600},
        "/api/ingest": {"requests": settings.api.INGEST_RATE_LIMIT, "window": 3600},
    }

# Pydantic models for API
class DocumentFilter(BaseModel):
    topic: Optional[str] = None
    source: Optional[str] = None
    doc_type: Optional[str] = None
    days: Optional[int] = 30
    search: Optional[str] = None
    limit: Optional[int] = 100

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    source_filter: Optional[str] = None
    doc_type_filter: Optional[str] = None
    k: Optional[int] = Field(default=8, ge=1, le=20)

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query_expansion: List[str]
    confidence: float
    processing_time: float

class IngestRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    days: Optional[int] = Field(default=30, ge=1, le=365)
    rebuild_index: Optional[bool] = False

class StatsResponse(BaseModel):
    total_documents: int
    by_source: Dict[str, int]
    by_doc_type: Dict[str, int]
    recent_activity: Dict[str, int]
    last_update: Optional[str]

# FastAPI app
app = FastAPI(
    title="Policy Radar API",
    description="Brussels public affairs platform with AI-enhanced document tracking",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Security middleware
if ADVANCED_CONFIG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.security.TRUSTED_HOSTS
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept", "Authorization"],
    )
else:
    # Fallback CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Advanced security headers and rate limiting
if ADVANCED_CONFIG:
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        # Apply rate limiting
        try:
            client_ip = request.client.host if request.client else "unknown"
            endpoint = request.url.path
            
            limit_config = RATE_LIMITS.get(endpoint, RATE_LIMITS["default"])
            
            with rate_limit_lock:
                now = time.time()
                client_key = f"{client_ip}:{endpoint}"
                
                if now > rate_limit_storage[client_key]["reset_time"]:
                    rate_limit_storage[client_key] = {
                        "count": 0,
                        "reset_time": now + limit_config["window"]
                    }
                
                if rate_limit_storage[client_key]["count"] >= limit_config["requests"]:
                    return Response(
                        content=f"Rate limit exceeded. Try again in {int(rate_limit_storage[client_key]['reset_time'] - now)} seconds.",
                        status_code=429,
                        headers={"Retry-After": str(int(rate_limit_storage[client_key]["reset_time"] - now))}
                    )
                
                rate_limit_storage[client_key]["count"] += 1
        except Exception as e:
            print(f"Rate limiting error: {e}")
        
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

# Global state
vector_store: Optional[PolicyVectorStore] = None
rag_service: Optional[PolicyRAGService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global vector_store, rag_service

    print("[INFO] Starting Policy Radar API...")

    # Advanced configuration validation
    if ADVANCED_CONFIG:
        try:
            validate_startup_config()
        except Exception as e:
            print(f"[ERROR] Configuration validation failed: {e}")
            if settings.is_production():
                raise e
            print("[WARN] Continuing in development mode")
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    VECTORS_DIR.mkdir(exist_ok=True)

    # Try to load RAG services if available
    if RAG_AVAILABLE and INDEX_PATH.exists():
        try:
            print("[INFO] Loading vector index and RAG service...")
            vector_store = PolicyVectorStore()
            vector_store.load_index(str(INDEX_PATH))
            rag_service = PolicyRAGService(str(INDEX_PATH))
            print("[OK] RAG services initialized")
        except Exception as e:
            print(f"[WARN] Could not load RAG services: {e}")
            print("RAG endpoints will use mock responses")
    else:
        print("[INFO] RAG services not available or no vector index found")

def load_documents_from_jsonl() -> List[Dict[str, Any]]:
    """Load documents from JSONL file or generate sample data"""
    jsonl_path = DATA_DIR / "items.jsonl"
    
    if jsonl_path.exists():
        documents = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        documents.append(json.loads(line))
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
    
    # Generate sample data if no real data exists
    sample_documents = []
    topics = ["Climate Change", "Digital Single Market", "Trade Policy", "Energy Transition", "Agriculture", "Migration"]
    sources = ["EURACTIV", "EUR-Lex", "European Parliament"]
    
    for i in range(50):
        doc = {
            "doc_id": f"doc_{i:03d}",
            "title": f"Sample EU Policy Document {i+1}",
            "summary": f"This is a sample policy document about {topics[i % len(topics)]}. It demonstrates the API functionality with realistic-looking data.",
            "content": f"Full content of document {i+1} would be here. This covers {topics[i % len(topics)]} policy implications...",
            "source": sources[i % len(sources)],
            "doc_type": "Policy Brief" if i % 3 == 0 else "Regulation" if i % 3 == 1 else "News Article",
            "published": (datetime.utcnow() - timedelta(days=i)).isoformat(),
            "url": f"https://example.com/document/{i}",
            "topics": [topics[i % len(topics)], topics[(i+1) % len(topics)]],
            "language": "en"
        }
        sample_documents.append(doc)
    
    return sample_documents

def filter_documents(documents: List[Dict], filters: DocumentFilter) -> List[Dict]:
    """Apply filters to document list"""
    filtered = documents.copy()

    # Date filter
    if filters.days:
        cutoff = datetime.utcnow() - timedelta(days=filters.days)
        filtered = [
            doc for doc in filtered 
            if doc.get('published') and 
            datetime.fromisoformat(doc['published'].replace('Z', '+00:00')).replace(tzinfo=None) >= cutoff
        ]

    # Topic filter
    if filters.topic and filters.topic != 'all':
        filtered = [
            doc for doc in filtered
            if any(filters.topic.lower() in topic.lower() for topic in doc.get('topics', []))
        ]

    # Source filter
    if filters.source and filters.source != 'all':
        filtered = [doc for doc in filtered if doc.get('source') == filters.source]

    # Document type filter
    if filters.doc_type and filters.doc_type != 'all':
        filtered = [doc for doc in filtered if doc.get('doc_type') == filters.doc_type]

    # Search filter
    if filters.search:
        search_term = filters.search.lower()
        filtered = [
            doc for doc in filtered
            if (search_term in doc.get('title', '').lower() or
                search_term in doc.get('summary', '').lower() or
                any(search_term in topic.lower() for topic in doc.get('topics', [])))
        ]

    # Sort by date (newest first)
    filtered.sort(key=lambda x: x.get('published', ''), reverse=True)

    # Limit results
    if filters.limit:
        filtered = filtered[:filters.limit]

    return filtered

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Policy Radar API - Enhanced Version",
        "version": "2.0.0",
        "description": "Brussels public affairs platform with AI-enhanced document tracking",
        "features": {
            "advanced_config": ADVANCED_CONFIG,
            "rag_available": RAG_AVAILABLE,
            "vector_search": rag_service is not None,
            "rate_limiting": ADVANCED_CONFIG,
            "security_headers": ADVANCED_CONFIG
        },
        "endpoints": {
            "health": "/health",
            "api_health": "/api/health", 
            "documents": "/api/documents",
            "stats": "/api/stats",
            "rag_query": "/api/rag/query",
            "ingest": "/api/ingest",
            "topics": "/api/topics",
            "sources": "/api/sources",
            "documentation": "/api/docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@app.get("/api/health")
async def api_health():
    """Detailed API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "features": {
            "advanced_config": ADVANCED_CONFIG,
            "rag_available": RAG_AVAILABLE,
            "vector_store_loaded": vector_store is not None,
            "rag_service_loaded": rag_service is not None
        },
        "data_status": {
            "documents_available": len(load_documents_from_jsonl()) > 0,
            "data_path": str(DATA_DIR),
            "vector_path": str(VECTORS_DIR)
        }
    }

@app.get("/api/documents")
async def get_documents(
    topic: Optional[str] = Query(None, description="Filter by topic"),
    source: Optional[str] = Query(None, description="Filter by source"),
    doc_type: Optional[str] = Query(None, description="Filter by document type"),
    days: Optional[int] = Query(30, description="Number of days to look back"),
    search: Optional[str] = Query(None, description="Search term"),
    limit: Optional[int] = Query(100, description="Maximum number of results")
):
    """Get policy documents with optional filters"""
    documents = load_documents_from_jsonl()

    if not documents:
        raise HTTPException(
            status_code=404, 
            detail="No documents found. Data ingestion may be needed."
        )

    filters = DocumentFilter(
        topic=topic,
        source=source, 
        doc_type=doc_type,
        days=days,
        search=search,
        limit=limit
    )

    filtered_docs = filter_documents(documents, filters)

    return {
        "documents": filtered_docs,
        "total": len(filtered_docs),
        "filters_applied": filters.dict(),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get dashboard statistics"""
    documents = load_documents_from_jsonl()

    if not documents:
        return StatsResponse(
            total_documents=0,
            by_source={},
            by_doc_type={},
            recent_activity={},
            last_update=None
        )

    # Calculate stats
    by_source = {}
    by_doc_type = {}
    recent_activity = {"last_7_days": 0, "last_30_days": 0}

    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    for doc in documents:
        # Source stats
        source = doc.get('source', 'Unknown')
        by_source[source] = by_source.get(source, 0) + 1

        # Doc type stats
        doc_type = doc.get('doc_type', 'Unknown')
        by_doc_type[doc_type] = by_doc_type.get(doc_type, 0) + 1

        # Recent activity
        if doc.get('published'):
            try:
                pub_date = datetime.fromisoformat(doc['published'].replace('Z', '+00:00')).replace(tzinfo=None)
                if pub_date >= week_ago:
                    recent_activity["last_7_days"] += 1
                if pub_date >= month_ago:
                    recent_activity["last_30_days"] += 1
            except:
                pass

    # Get last update time
    last_update = None
    jsonl_path = DATA_DIR / "items.jsonl"
    if jsonl_path.exists():
        last_update = datetime.fromtimestamp(jsonl_path.stat().st_mtime).isoformat()

    return StatsResponse(
        total_documents=len(documents),
        by_source=by_source,
        by_doc_type=by_doc_type,
        recent_activity=recent_activity,
        last_update=last_update
    )

@app.post("/api/rag/query", response_model=RAGResponse)
async def rag_query(query_request: RAGQuery, request: Request):
    """Submit a RAG query for natural language Q&A"""
    # Input validation
    query_text = query_request.query.strip()
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Security validation
    max_length = settings.security.MAX_QUERY_LENGTH if ADVANCED_CONFIG else 500
    if len(query_text) > max_length:
        raise HTTPException(
            status_code=400, 
            detail=f"Query too long. Maximum {max_length} characters allowed."
        )
    
    # Check for suspicious patterns
    suspicious_patterns = ["<script", "javascript:", "eval(", "exec(", "__import__"]
    if any(pattern in query_text.lower() for pattern in suspicious_patterns):
        raise HTTPException(status_code=400, detail="Invalid query format")

    start_time = datetime.utcnow()

    try:
        if rag_service and RAG_AVAILABLE:
            # Use real RAG service
            result = rag_service.query(
                query_text,
                source_filter=query_request.source_filter,
                doc_type_filter=query_request.doc_type_filter,
                k=min(query_request.k, 10)
            )
            
            # Convert sources to serializable format
            sources = []
            for chunk in result.sources:
                sources.append({
                    "chunk_id": getattr(chunk, 'chunk_id', 'unknown'),
                    "doc_id": getattr(chunk, 'doc_id', 'unknown'),
                    "source": getattr(chunk, 'source', 'unknown'),
                    "doc_type": getattr(chunk, 'doc_type', 'unknown'),
                    "title": getattr(chunk, 'title', 'Untitled'),
                    "content": getattr(chunk, 'content', '')[:200] + "..." if len(getattr(chunk, 'content', '')) > 200 else getattr(chunk, 'content', ''),
                    "url": getattr(chunk, 'url', ''),
                    "published": getattr(chunk, 'published', ''),
                    "topics": getattr(chunk, 'topics', [])
                })
            
            answer = result.answer
            query_expansion = result.query_expansion
            confidence = result.confidence
        else:
            # Use mock response
            documents = load_documents_from_jsonl()
            relevant_docs = [doc for doc in documents[:3]]  # Mock: first 3 docs
            
            sources = []
            for doc in relevant_docs:
                sources.append({
                    "chunk_id": f"chunk_{doc.get('doc_id', 'unknown')}",
                    "doc_id": doc.get('doc_id', 'unknown'),
                    "source": doc.get('source', 'unknown'),
                    "doc_type": doc.get('doc_type', 'unknown'),
                    "title": doc.get('title', 'Untitled'),
                    "content": doc.get('summary', '')[:200] + "...",
                    "url": doc.get('url', ''),
                    "published": doc.get('published', ''),
                    "topics": doc.get('topics', [])
                })
            
            answer = f"Based on the available documents, here's what I found about '{query_text}': This is a mock response demonstrating the RAG functionality. In production, this would provide AI-generated insights based on the retrieved policy documents."
            query_expansion = [query_text, query_text.replace(' ', '_')]
            confidence = 0.75

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return RAGResponse(
            answer=answer,
            sources=sources,
            query_expansion=query_expansion,
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing RAG query: {str(e)}"
        )

@app.post("/api/ingest") 
async def trigger_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger data ingestion and vector indexing"""
    
    # Input validation
    topic = request.topic.strip()
    max_length = settings.security.MAX_TOPIC_LENGTH if ADVANCED_CONFIG else 50
    
    if not topic or len(topic) < 2 or len(topic) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Topic must be between 2 and {max_length} characters"
        )
    
    # Validate topic format
    import re
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', topic):
        raise HTTPException(
            status_code=400,
            detail="Topic contains invalid characters"
        )
    
    # Mock ingestion response (in production, would trigger background task)
    return {
        "message": f"Data ingestion started for topic: {topic}",
        "status": "started",
        "estimated_duration": "2-5 minutes",
        "check_status": "/api/health",
        "parameters": {
            "topic": topic,
            "days": request.days,
            "rebuild_index": request.rebuild_index
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/topics")
async def get_topics():
    """Get available topics from current dataset"""
    documents = load_documents_from_jsonl()

    all_topics = set()
    for doc in documents:
        all_topics.update(doc.get('topics', []))

    # Count documents per topic
    topic_counts = {}
    for topic in all_topics:
        count = sum(1 for doc in documents if topic in doc.get('topics', []))
        topic_counts[topic] = count

    # Sort by popularity
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "topics": [{"name": topic, "count": count} for topic, count in sorted_topics],
        "total_unique_topics": len(all_topics),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/sources")
async def get_sources():
    """Get available sources from current dataset"""
    documents = load_documents_from_jsonl()

    source_counts = {}
    for doc in documents:
        source = doc.get('source', 'Unknown')
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "sources": [{"name": source, "count": count} for source, count in source_counts.items()],
        "total_sources": len(source_counts),
        "timestamp": datetime.utcnow().isoformat()
    }

# Development and production server
if __name__ == "__main__":
    import os
    # Railway PORT override or use settings
    if ADVANCED_CONFIG:
        port = int(os.getenv("PORT", settings.api.PORT))
        host = settings.api.HOST
        reload = settings.is_development()
        workers = 1  # Railway uses single worker
    else:
        port = int(os.getenv("PORT", 8000))
        host = "0.0.0.0"
        reload = os.getenv("ENVIRONMENT", "development") != "production"
        workers = 1
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )
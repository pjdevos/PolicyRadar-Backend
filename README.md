# Policy Radar API - Railway Deployment

Clean, minimal FastAPI application for Railway deployment.

Last updated: 2025-08-19 15:20 UTC

## Files

- `main.py` - FastAPI application
- `requirements.txt` - Python dependencies  
- `Procfile` - Railway start command
- `runtime.txt` - Python version specification

## Endpoints

### Core
- `/` - API info with features and endpoint overview
- `/health` & `/api/health` - Unified health check with detailed status

### Data Access  
- `/api/documents` - Policy documents with filtering (topic, source, doc_type, days, search, limit)
- `/api/stats` - Dashboard statistics (document counts, sources, recent activity)
- `/api/topics` - Available topics with document counts
- `/api/sources` - Available sources with document counts

### AI Services
- `/api/rag/query` - RAG-powered natural language Q&A over policy documents
- `/api/ingest` - Trigger data ingestion and vector indexing

### Documentation
- `/api/docs` - Interactive OpenAPI documentation
- `/api/redoc` - ReDoc API documentation

## Deployment

This project is configured for Railway deployment with automatic detection.
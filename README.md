# Policy Radar API - Railway Deployment

Clean, minimal FastAPI application for Railway deployment.

Last updated: 2025-08-19

## Files

- `main.py` - FastAPI application
- `requirements.txt` - Python dependencies  
- `Procfile` - Railway start command
- `runtime.txt` - Python version specification

## Endpoints

- `/` - API info
- `/health` - Health check
- `/api/health` - API health check
- `/api/status` - Service status

## Deployment

This project is configured for Railway deployment with automatic detection.
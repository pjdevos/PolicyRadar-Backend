#!/usr/bin/env python3
"""
Policy Radar API - Clean Railway Deployment
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Policy Radar API",
    description="Brussels public affairs platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "name": "Policy Radar API",
        "version": "1.0.0",
        "status": "running",
        "message": "Brussels public affairs platform"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/health")
def api_health():
    return {"status": "healthy", "service": "Policy Radar API"}

@app.get("/api/status")
def api_status():
    return {
        "status": "operational",
        "service": "Policy Radar API",
        "endpoints": {
            "health": "/health",
            "api_health": "/api/health",
            "status": "/api/status"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Policy Radar API on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
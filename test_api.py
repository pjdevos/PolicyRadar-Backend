#!/usr/bin/env python3
"""
Test script for Policy Radar Railway API
"""
import requests
import json

# Replace with your actual Railway URL
BASE_URL = "https://your-app.railway.app"  # Update this!

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n🔍 Testing {method} {endpoint}")
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Response: {json.dumps(result, indent=2)[:500]}...")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    print("🚀 Testing Policy Radar API on Railway")
    print("="*50)
    
    # Test basic endpoints
    test_endpoint("/")
    test_endpoint("/health")
    test_endpoint("/api/health")
    
    # Test data endpoints
    test_endpoint("/api/documents")
    test_endpoint("/api/stats")
    test_endpoint("/api/topics")
    test_endpoint("/api/sources")
    
    # Test filtering
    test_endpoint("/api/documents?topic=hydrogen")
    test_endpoint("/api/documents?source=EUR-Lex")
    test_endpoint("/api/documents?search=transport")
    
    # Test RAG query
    rag_data = {"query": "What are the latest developments in hydrogen policy?"}
    test_endpoint("/api/rag/query", method="POST", data=rag_data)
    
    print("\n" + "="*50)
    print("✅ API testing complete!")
    print("Update BASE_URL with your actual Railway URL to test")

if __name__ == "__main__":
    main()
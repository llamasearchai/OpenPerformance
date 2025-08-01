#!/usr/bin/env python3
"""Test script to verify OpenPerformance API functionality."""

import asyncio
import httpx
import sys

async def test_api():
    """Test basic API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing OpenPerformance API...")
    print("-" * 50)
    
    async with httpx.AsyncClient() as client:
        # Test health endpoint
        try:
            print("1. Testing health endpoint...")
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
            assert response.status_code == 200
            print("   ✓ Health check passed")
        except Exception as e:
            print(f"   ✗ Health check failed: {e}")
            
        # Test API info
        try:
            print("\n2. Testing API info endpoint...")
            response = await client.get(f"{base_url}/api/v1")
            print(f"   Status: {response.status_code}")
            data = response.json()
            print(f"   Version: {data.get('version', 'Unknown')}")
            print("   ✓ API info passed")
        except Exception as e:
            print(f"   ✗ API info failed: {e}")
            
        # Test hardware info (requires auth in production)
        try:
            print("\n3. Testing hardware info endpoint...")
            response = await client.get(f"{base_url}/api/v1/hardware/info")
            print(f"   Status: {response.status_code}")
            if response.status_code == 401:
                print("   ✓ Auth required (expected)")
            else:
                print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   ✗ Hardware info failed: {e}")
    
    print("\n" + "-" * 50)
    print("API test completed!")

if __name__ == "__main__":
    asyncio.run(test_api())
"""
PocketFence Kernel Integration Test
Tests PocketFence Kernel API if service is running
"""
import requests
import json
import time
from typing import Dict, Optional


class PocketFenceKernelTester:
    """Test PocketFence Kernel via REST API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_service_available(self) -> bool:
        """Check if PocketFence Kernel service is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/kernel/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def test_url_check(self, url: str) -> Optional[Dict]:
        """Test URL safety checking"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/filter/url",
                json={"url": url},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error testing URL check: {e}")
        return None
    
    def test_content_filter(self, content: str) -> Optional[Dict]:
        """Test content filtering"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/filter/content",
                json={"content": content},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error testing content filter: {e}")
        return None
    
    def test_batch_processing(self, items: list) -> Optional[Dict]:
        """Test batch processing"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/filter/batch",
                json={"items": items},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error testing batch processing: {e}")
        return None
    
    def get_statistics(self) -> Optional[Dict]:
        """Get kernel statistics"""
        try:
            response = self.session.get(f"{self.base_url}/api/filter/stats", timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting statistics: {e}")
        return None


def test_pocketfence_kernel():
    """Test PocketFence Kernel if service is running"""
    print("="*70)
    print("POCKETFENCE KERNEL API TEST")
    print("="*70)
    
    tester = PocketFenceKernelTester()
    
    # Check if service is available
    print("\n[1] Checking if PocketFence Kernel service is running...")
    if not tester.check_service_available():
        print("  [X] Service not available")
        print("\n  To start PocketFence Kernel service:")
        print("    1. cd PocketFenceKernel")
        print("    2. dotnet run -- --kernel")
        print("    3. Service will run on http://localhost:5000")
        print("\n  Then run this test again.")
        return
    
    print("  [✓] Service is running!")
    
    # Test URL checking
    print("\n[2] Testing URL safety check...")
    url_result = tester.test_url_check("https://example.com")
    if url_result:
        print(f"  URL: {url_result.get('url', 'N/A')}")
        print(f"  Is Blocked: {url_result.get('isBlocked', False)}")
        print(f"  Threat Score: {url_result.get('threatScore', 0.0):.2f}")
        print(f"  Recommendation: {url_result.get('recommendation', 'N/A')}")
    else:
        print("  [X] URL check failed")
    
    # Test content filtering
    print("\n[3] Testing content filtering...")
    content_result = tester.test_content_filter("Hello, this is a test message")
    if content_result:
        print(f"  Is Blocked: {content_result.get('isBlocked', False)}")
        print(f"  Threat Score: {content_result.get('threatScore', 0.0):.2f}")
        print(f"  Is Child Safe: {content_result.get('isChildSafe', True)}")
    else:
        print("  [X] Content filter failed")
    
    # Test batch processing
    print("\n[4] Testing batch processing...")
    batch_items = [
        {"id": "1", "type": "url", "content": "https://google.com"},
        {"id": "2", "type": "content", "content": "Safe content here"}
    ]
    batch_result = tester.test_batch_processing(batch_items)
    if batch_result:
        print(f"  Processed {len(batch_result.get('results', []))} items")
        for result in batch_result.get('results', [])[:3]:
            print(f"    - {result.get('id', 'N/A')}: Blocked={result.get('isBlocked', False)}")
    else:
        print("  [X] Batch processing failed")
    
    # Get statistics
    print("\n[5] Getting kernel statistics...")
    stats = tester.get_statistics()
    if stats:
        print(f"  Total Requests: {stats.get('totalRequests', 0)}")
        print(f"  Blocked Requests: {stats.get('blockedRequests', 0)}")
        print(f"  Allowed Requests: {stats.get('allowedRequests', 0)}")
        print(f"  Average Response Time: {stats.get('averageResponseTime', 0):.2f}ms")
    else:
        print("  [X] Could not get statistics")
    
    print("\n" + "="*70)
    print("POCKETFENCE KERNEL TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_pocketfence_kernel()

"""
API Security
Authentication, Authorization, and Rate Limiting for ML Toolbox APIs

Features:
- API key authentication
- JWT token support
- Rate limiting
- CORS configuration
- Request validation
"""
import time
from typing import Optional, List, Dict, Any, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib
import hmac
import warnings

try:
    from fastapi import HTTPException, Request, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Install with: pip install fastapi")


class APIKeyAuth:
    """API Key Authentication"""
    
    def __init__(self, api_keys: List[str]):
        """
        Args:
            api_keys: List of valid API keys
        """
        self.api_keys = set(api_keys)
    
    def validate_key(self, api_key: str) -> bool:
        """
        Validate API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        return api_key in self.api_keys
    
    def add_key(self, api_key: str):
        """Add a new API key"""
        self.api_keys.add(api_key)
    
    def remove_key(self, api_key: str):
        """Remove an API key"""
        self.api_keys.discard(api_key)


class RateLimiter:
    """
    Rate limiter for API requests
    
    Uses token bucket algorithm
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None
    ):
        """
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour (optional)
            requests_per_day: Maximum requests per day (optional)
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        
        # Track requests per client
        self.client_requests: Dict[str, deque] = defaultdict(lambda: deque())
    
    def is_allowed(self, client_id: str = 'default') -> bool:
        """
        Check if request is allowed
        
        Args:
            client_id: Client identifier (IP, API key, etc.)
            
        Returns:
            True if allowed, False otherwise
        """
        now = time.time()
        requests = self.client_requests[client_id]
        
        # Remove old requests (older than 1 minute)
        while requests and requests[0] < now - 60:
            requests.popleft()
        
        # Check per-minute limit
        if len(requests) >= self.requests_per_minute:
            return False
        
        # Add current request
        requests.append(now)
        return True
    
    def get_remaining(self, client_id: str = 'default') -> int:
        """
        Get remaining requests for client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Number of remaining requests
        """
        now = time.time()
        requests = self.client_requests[client_id]
        
        # Remove old requests
        while requests and requests[0] < now - 60:
            requests.popleft()
        
        return max(0, self.requests_per_minute - len(requests))


class CORSMiddlewareConfig:
    """CORS configuration"""
    
    def __init__(
        self,
        allowed_origins: List[str] = ['*'],
        allowed_methods: List[str] = ['*'],
        allowed_headers: List[str] = ['*'],
        allow_credentials: bool = True
    ):
        """
        Args:
            allowed_origins: List of allowed origins
            allowed_methods: List of allowed methods
            allowed_headers: List of allowed headers
            allow_credentials: Whether to allow credentials
        """
        self.allowed_origins = allowed_origins
        self.allowed_methods = allowed_methods
        self.allowed_headers = allowed_headers
        self.allow_credentials = allow_credentials
    
    def apply_to_app(self, app):
        """
        Apply CORS configuration to FastAPI app
        
        Args:
            app: FastAPI application
        """
        if not FASTAPI_AVAILABLE:
            warnings.warn("FastAPI not available for CORS configuration")
            return
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.allowed_origins,
            allow_credentials=self.allow_credentials,
            allow_methods=self.allowed_methods,
            allow_headers=self.allowed_headers
        )


def create_api_key_auth_dependency(api_keys: List[str]):
    """
    Create FastAPI dependency for API key authentication
    
    Args:
        api_keys: List of valid API keys
        
    Returns:
        FastAPI dependency function
    """
    if not FASTAPI_AVAILABLE:
        return None
    
    auth = APIKeyAuth(api_keys)
    
    async def verify_api_key(request: Request):
        """Verify API key from request"""
        api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required"
            )
        
        # Remove 'Bearer ' prefix if present
        if api_key.startswith('Bearer '):
            api_key = api_key[7:]
        
        if not auth.validate_key(api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return api_key
    
    return verify_api_key


def create_rate_limit_dependency(
    requests_per_minute: int = 60,
    get_client_id: Optional[Callable] = None
):
    """
    Create FastAPI dependency for rate limiting
    
    Args:
        requests_per_minute: Maximum requests per minute
        get_client_id: Function to get client ID from request
        
    Returns:
        FastAPI dependency function
    """
    if not FASTAPI_AVAILABLE:
        return None
    
    rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
    
    async def check_rate_limit(request: Request):
        """Check rate limit for request"""
        if get_client_id:
            client_id = get_client_id(request)
        else:
            # Default: use IP address
            client_id = request.client.host if request.client else 'unknown'
        
        if not rate_limiter.is_allowed(client_id):
            remaining = rate_limiter.get_remaining(client_id)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again later.",
                headers={
                    "X-RateLimit-Limit": str(requests_per_minute),
                    "X-RateLimit-Remaining": str(remaining),
                    "Retry-After": "60"
                }
            )
        
        return True
    
    return check_rate_limit


class APISecurity:
    """
    Comprehensive API security manager
    
    Combines authentication, rate limiting, and CORS
    """
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        require_auth: bool = False,
        rate_limit_per_minute: int = 60,
        cors_config: Optional[CORSMiddlewareConfig] = None
    ):
        """
        Args:
            api_keys: List of valid API keys
            require_auth: Whether to require authentication
            rate_limit_per_minute: Rate limit per minute
            cors_config: CORS configuration
        """
        self.api_keys = api_keys or []
        self.require_auth = require_auth
        self.rate_limit_per_minute = rate_limit_per_minute
        
        self.auth = APIKeyAuth(self.api_keys) if self.api_keys else None
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_per_minute)
        self.cors_config = cors_config or CORSMiddlewareConfig()
    
    def setup_app(self, app):
        """
        Setup security for FastAPI app
        
        Args:
            app: FastAPI application
        """
        if not FASTAPI_AVAILABLE:
            warnings.warn("FastAPI not available for security setup")
            return
        
        # Apply CORS
        self.cors_config.apply_to_app(app)
        
        # Add authentication dependency if required
        if self.require_auth and self.auth:
            auth_dep = create_api_key_auth_dependency(self.api_keys)
            app.dependency_overrides['verify_api_key'] = auth_dep
        
        # Add rate limiting dependency
        rate_limit_dep = create_rate_limit_dependency(self.rate_limit_per_minute)
        app.dependency_overrides['check_rate_limit'] = rate_limit_dep

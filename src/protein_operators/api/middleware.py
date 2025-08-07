"""
Middleware for API request processing and security.
"""

import time
import json
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track request processing time."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log slow requests
        if process_time > 10.0:  # Log requests taking more than 10 seconds
            logger.warning(
                f"Slow request: {request.method} {request.url.path} took {process_time:.2f}s"
            )
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log API requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            f"Request: {request.method} {request.url.path} from {client_ip}"
        )
        
        # Add request ID for tracing
        request_id = f"req_{int(time.time() * 1000)}"
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            
            # Log response
            logger.info(
                f"Response: {request_id} - {response.status_code} "
                f"for {request.method} {request.url.path}"
            )
            
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            logger.error(
                f"Error in request {request_id}: {str(e)} "
                f"for {request.method} {request.url.path}"
            )
            raise


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.client_calls: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Initialize client tracking
        if client_ip not in self.client_calls:
            self.client_calls[client_ip] = []
        
        # Clean old calls (older than 1 minute)
        self.client_calls[client_ip] = [
            call_time for call_time in self.client_calls[client_ip]
            if current_time - call_time < 60
        ]
        
        # Check rate limit
        if len(self.client_calls[client_ip]) >= self.calls_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls_per_minute} requests per minute"
                }
            )
        
        # Record this call
        self.client_calls[client_ip].append(current_time)
        
        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to handle and format errors consistently."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
            
        except HTTPException:
            # Re-raise HTTP exceptions (they're handled by FastAPI)
            raise
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)} for {request.url.path}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Validation Error",
                    "message": str(e),
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)} for {request.url.path}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Resource Not Found",
                    "message": str(e),
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)} for {request.url.path}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Simple JWT authentication middleware."""
    
    def __init__(self, app, secret_key: str = "your-secret-key", require_auth: bool = False):
        super().__init__(app)
        self.secret_key = secret_key
        self.require_auth = require_auth
        
        # Endpoints that don't require authentication
        self.public_endpoints = {
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)
        
        # Skip if authentication is not required
        if not self.require_auth:
            return await call_next(request)
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication Required",
                    "message": "Missing or invalid Authorization header"
                }
            )
        
        token = auth_header.split("Bearer ")[1]
        
        try:
            # Verify JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            request.state.user = payload
            
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Token Expired",
                    "message": "The provided token has expired"
                }
            )
            
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid Token",
                    "message": "The provided token is invalid"
                }
            )
        
        return await call_next(request)


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware for cross-origin requests."""
    
    def __init__(
        self,
        app,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight OPTIONS request
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*" if "*" in self.allow_origins else ", ".join(self.allow_origins),
                    "Access-Control-Allow-Methods": ", ".join(self.allow_methods),
                    "Access-Control-Allow-Headers": ", ".join(self.allow_headers),
                    "Access-Control-Max-Age": "600"
                }
            )
        
        response = await call_next(request)
        
        # Add CORS headers to response
        response.headers["Access-Control-Allow-Origin"] = "*" if "*" in self.allow_origins else ", ".join(self.allow_origins)
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Simple response compression middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        
        if "gzip" in accept_encoding and response.headers.get("Content-Type", "").startswith(("application/json", "text/")):
            # In a real implementation, you would compress the response body here
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


def create_token(user_data: Dict[str, Any], secret_key: str, expires_delta: timedelta = None) -> str:
    """
    Create a JWT token for authentication.
    
    Args:
        user_data: User information to encode in token
        secret_key: Secret key for signing
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(hours=24)
    
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        **user_data,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    return jwt.encode(payload, secret_key, algorithm="HS256")


def verify_token(token: str, secret_key: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
        secret_key: Secret key for verification
        
    Returns:
        Decoded token payload
        
    Raises:
        jwt.InvalidTokenError: If token is invalid or expired
    """
    return jwt.decode(token, secret_key, algorithms=["HS256"])
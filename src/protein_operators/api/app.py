"""
FastAPI application factory and configuration.

Creates and configures the FastAPI application for protein design services.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
import uvicorn

from protein_operators.database import get_database, run_migrations
from protein_operators.core import ProteinDesigner
from .routes import router
from .middleware import setup_middleware
from .models import ErrorResponse

logger = logging.getLogger(__name__)

# Global application instance
_app_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Protein Operators API...")
    
    try:
        # Initialize database
        db = get_database()
        db.create_tables()
        logger.info("Database initialized successfully")
        
        # Run any pending migrations
        run_migrations()
        logger.info("Database migrations completed")
        
        # Initialize global protein designer
        app.state.designer = ProteinDesigner(
            operator_type=os.getenv("DEFAULT_OPERATOR_TYPE", "deeponet"),
            device=os.getenv("CUDA_VISIBLE_DEVICES", "auto")
        )
        logger.info("Protein designer initialized")
        
        # Store database reference
        app.state.database = db
        
        logger.info("Protein Operators API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Protein Operators API...")
    
    try:
        # Close database connections
        if hasattr(app.state, 'database'):
            app.state.database.close()
            logger.info("Database connections closed")
        
        # Cleanup other resources
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app(
    title: str = "Protein Operators API",
    description: str = "RESTful API for zero-shot protein design",
    version: str = "1.0.0",
    debug: bool = False,
    **kwargs
) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        title: API title
        description: API description
        version: API version
        debug: Debug mode flag
        **kwargs: Additional FastAPI arguments
        
    Returns:
        Configured FastAPI application
    """
    
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
        **kwargs
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE").split(","),
        allow_headers=os.getenv("ALLOWED_HEADERS", "*").split(","),
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": title,
            "version": version,
            "description": description,
            "docs": "/docs" if debug else "Documentation available in debug mode",
            "status": "healthy"
        }
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        try:
            # Check database connection
            db = get_database()
            with db.session() as session:
                session.execute("SELECT 1")
            
            return {
                "status": "healthy",
                "database": "connected",
                "api": "operational"
            }
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e)
                }
            )
    
    # Global exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error for {request.url}: {exc}")
        
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="Validation Error",
                message="Request validation failed",
                details=exc.errors()
            ).dict()
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP error {exc.status_code} for {request.url}: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"HTTP {exc.status_code}",
                message=exc.detail
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled error for {request.url}: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                message="An unexpected error occurred"
            ).dict()
        )
    
    return app


def get_app() -> FastAPI:
    """Get or create global application instance."""
    global _app_instance
    
    if _app_instance is None:
        _app_instance = create_app(
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    return _app_instance


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    **kwargs
):
    """Run the API server with uvicorn."""
    
    # Get configuration from environment
    host = os.getenv("API_HOST", host)
    port = int(os.getenv("API_PORT", port))
    workers = int(os.getenv("API_WORKERS", workers))
    reload = os.getenv("API_RELOAD", str(reload)).lower() == "true"
    
    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    
    # Configure uvicorn
    config = {
        "app": "protein_operators.api.app:get_app",
        "factory": True,
        "host": host,
        "port": port,
        "reload": reload,
        "access_log": True,
        "log_level": "info",
        **kwargs
    }
    
    if workers > 1 and not reload:
        config["workers"] = workers
    
    uvicorn.run(**config)


if __name__ == "__main__":
    # Run server if script is executed directly
    run_server(reload=True)
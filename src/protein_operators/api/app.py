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
    @app.get("/", tags=["Root"])\n    async def root():\n        \"\"\"Root endpoint with API information.\"\"\"\n        return {\n            \"name\": title,\n            \"version\": version,\n            \"description\": description,\n            \"docs\": \"/docs\" if debug else \"Documentation available in debug mode\",\n            \"status\": \"healthy\"\n        }\n    \n    # Health check endpoint\n    @app.get(\"/health\", tags=[\"Health\"])\n    async def health_check():\n        \"\"\"Health check endpoint.\"\"\"\n        try:\n            # Check database connection\n            db = get_database()\n            with db.session() as session:\n                session.execute(\"SELECT 1\")\n            \n            return {\n                \"status\": \"healthy\",\n                \"database\": \"connected\",\n                \"api\": \"operational\"\n            }\n        \n        except Exception as e:\n            logger.error(f\"Health check failed: {e}\")\n            return JSONResponse(\n                status_code=503,\n                content={\n                    \"status\": \"unhealthy\",\n                    \"error\": str(e)\n                }\n            )\n    \n    # Global exception handlers\n    @app.exception_handler(RequestValidationError)\n    async def validation_exception_handler(request: Request, exc: RequestValidationError):\n        \"\"\"Handle request validation errors.\"\"\"\n        logger.warning(f\"Validation error for {request.url}: {exc}\")\n        \n        return JSONResponse(\n            status_code=422,\n            content=ErrorResponse(\n                error=\"Validation Error\",\n                message=\"Request validation failed\",\n                details=exc.errors()\n            ).dict()\n        )\n    \n    @app.exception_handler(HTTPException)\n    async def http_exception_handler(request: Request, exc: HTTPException):\n        \"\"\"Handle HTTP exceptions.\"\"\"\n        logger.warning(f\"HTTP error {exc.status_code} for {request.url}: {exc.detail}\")\n        \n        return JSONResponse(\n            status_code=exc.status_code,\n            content=ErrorResponse(\n                error=f\"HTTP {exc.status_code}\",\n                message=exc.detail\n            ).dict()\n        )\n    \n    @app.exception_handler(Exception)\n    async def general_exception_handler(request: Request, exc: Exception):\n        \"\"\"Handle general exceptions.\"\"\"\n        logger.error(f\"Unhandled error for {request.url}: {exc}\", exc_info=True)\n        \n        return JSONResponse(\n            status_code=500,\n            content=ErrorResponse(\n                error=\"Internal Server Error\",\n                message=\"An unexpected error occurred\"\n            ).dict()\n        )\n    \n    return app\n\n\ndef get_app() -> FastAPI:\n    \"\"\"Get or create global application instance.\"\"\"\n    global _app_instance\n    \n    if _app_instance is None:\n        _app_instance = create_app(\n            debug=os.getenv(\"DEBUG\", \"false\").lower() == \"true\"\n        )\n    \n    return _app_instance\n\n\ndef run_server(\n    host: str = \"0.0.0.0\",\n    port: int = 8000,\n    reload: bool = False,\n    workers: int = 1,\n    **kwargs\n):\n    \"\"\"Run the API server with uvicorn.\"\"\"\n    \n    # Get configuration from environment\n    host = os.getenv(\"API_HOST\", host)\n    port = int(os.getenv(\"API_PORT\", port))\n    workers = int(os.getenv(\"API_WORKERS\", workers))\n    reload = os.getenv(\"API_RELOAD\", str(reload)).lower() == \"true\"\n    \n    logger.info(f\"Starting server on {host}:{port} with {workers} workers\")\n    \n    # Configure uvicorn\n    config = {\n        \"app\": \"protein_operators.api.app:get_app\",\n        \"factory\": True,\n        \"host\": host,\n        \"port\": port,\n        \"reload\": reload,\n        \"access_log\": True,\n        \"log_level\": \"info\",\n        **kwargs\n    }\n    \n    if workers > 1 and not reload:\n        config[\"workers\"] = workers\n    \n    uvicorn.run(**config)\n\n\nif __name__ == \"__main__\":\n    # Run server if script is executed directly\n    run_server(reload=True)"
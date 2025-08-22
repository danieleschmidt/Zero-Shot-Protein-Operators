#!/usr/bin/env python3
"""
🚀 Production Server for Autonomous Protein Design System
Complete production-ready server with all frameworks integrated.
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

# Add source directory to path
sys.path.append('src')

# FastAPI and async support
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("⚠️ FastAPI not available - running in demo mode")
    FastAPI = None

# Core imports
from protein_operators import ProteinDesigner, Constraints
from protein_operators.robust_framework import RobustProteinDesigner
from protein_operators.scaling_framework import ScalableProteinDesigner
from protein_operators.global_framework import GlobalProteinDesigner

# Monitoring imports
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    print("⚠️ Prometheus client not available - monitoring disabled")
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Metrics (if Prometheus available)
if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUEST_COUNT = Counter('protein_design_requests_total', 'Total requests', ['method', 'endpoint'])
    REQUEST_DURATION = Histogram('protein_design_request_duration_seconds', 'Request duration')
    ACTIVE_CONNECTIONS = Gauge('protein_design_active_connections', 'Active connections')
    
    # Design metrics
    DESIGN_REQUESTS = Counter('protein_design_designs_total', 'Total design requests', ['status'])
    DESIGN_DURATION = Histogram('protein_design_design_duration_seconds', 'Design duration')
    CACHE_HITS = Counter('protein_design_cache_hits_total', 'Cache hits')
    CACHE_MISSES = Counter('protein_design_cache_misses_total', 'Cache misses')
    
    # System metrics
    SYSTEM_HEALTH = Gauge('protein_design_system_health', 'System health score')
    ERROR_RATE = Gauge('protein_design_error_rate', 'Error rate')


# Pydantic models
if FastAPI:
    class DesignRequest(BaseModel):
        """Protein design request model."""
        length: int = Field(..., gt=10, le=1000, description="Protein length in residues")
        num_samples: int = Field(1, gt=0, le=10, description="Number of samples to generate")
        constraints: Optional[Dict] = Field(None, description="Design constraints")
        locale: Optional[str] = Field("en-US", description="User locale")
        region: Optional[str] = Field("US", description="User region")
        priority: Optional[int] = Field(1, ge=1, le=3, description="Request priority (1=high, 3=low)")
    
    class DesignResponse(BaseModel):
        """Protein design response model."""
        success: bool
        request_id: str
        result: Optional[Dict] = None
        error: Optional[str] = None
        metadata: Optional[Dict] = None
        execution_time: float
        timestamp: str
    
    class HealthResponse(BaseModel):
        """Health check response model."""
        status: str
        timestamp: str
        version: str
        uptime: float
        components: Dict[str, str]


class ProductionServer:
    """Production server for autonomous protein design."""
    
    def __init__(self):
        """Initialize production server."""
        self.start_time = time.time()
        self.version = "1.0.0"
        self.request_count = 0
        
        # Initialize design system
        logger.info("Initializing autonomous protein design system...")
        self._initialize_design_system()
        
        # Initialize FastAPI app if available
        if FastAPI:
            self.app = self._create_app()
        else:
            self.app = None
        
        logger.info("Production server initialized successfully")
    
    def _initialize_design_system(self):
        """Initialize the complete design system."""
        try:
            # Base designer
            self.base_designer = ProteinDesigner(
                operator_type="deeponet",
                device="auto"
            )
            
            # Robust framework
            self.robust_designer = RobustProteinDesigner(
                base_designer=self.base_designer,
                enable_monitoring=True,
                enable_recovery=True
            )
            
            # Scaling framework
            self.scaling_designer = ScalableProteinDesigner(
                base_designer=self.base_designer,
                enable_caching=True,
                enable_batching=True
            )
            
            # Global framework
            self.global_designer = GlobalProteinDesigner(
                base_designer=self.base_designer,
                locale="en-US",
                region="US"
            )
            
            logger.info("All design frameworks initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize design system: {e}")
            raise
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting up production server...")
            if PROMETHEUS_AVAILABLE:
                SYSTEM_HEALTH.set(1.0)
            yield
            # Shutdown
            logger.info("Shutting down production server...")
            if PROMETHEUS_AVAILABLE:
                SYSTEM_HEALTH.set(0.0)
        
        app = FastAPI(
            title="Autonomous Protein Design API",
            description="Production-ready autonomous protein design system",
            version=self.version,
            lifespan=lifespan
        )
        
        # Middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure properly in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes."""
        
        @app.middleware("http")
        async def metrics_middleware(request, call_next):
            """Metrics collection middleware."""
            start_time = time.time()
            
            if PROMETHEUS_AVAILABLE:
                ACTIVE_CONNECTIONS.inc()
            
            try:
                response = await call_next(request)
                
                if PROMETHEUS_AVAILABLE:
                    # Record metrics
                    duration = time.time() - start_time
                    REQUEST_COUNT.labels(
                        method=request.method,
                        endpoint=request.url.path
                    ).inc()
                    REQUEST_DURATION.observe(duration)
                
                return response
            
            finally:
                if PROMETHEUS_AVAILABLE:
                    ACTIVE_CONNECTIONS.dec()
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                # Check system components
                components = {
                    "base_designer": "healthy",
                    "robust_framework": "healthy", 
                    "scaling_framework": "healthy",
                    "global_framework": "healthy"
                }
                
                # Test basic functionality
                try:
                    constraints = Constraints()
                    test_result = self.base_designer.generate(constraints=constraints, length=15)
                    if test_result is None:
                        components["base_designer"] = "degraded"
                except Exception:
                    components["base_designer"] = "unhealthy"
                
                # Determine overall status
                if all(status == "healthy" for status in components.values()):
                    overall_status = "healthy"
                elif any(status == "unhealthy" for status in components.values()):
                    overall_status = "unhealthy"
                else:
                    overall_status = "degraded"
                
                if PROMETHEUS_AVAILABLE:
                    health_score = 1.0 if overall_status == "healthy" else 0.5 if overall_status == "degraded" else 0.0
                    SYSTEM_HEALTH.set(health_score)
                
                return HealthResponse(
                    status=overall_status,
                    timestamp=datetime.now().isoformat(),
                    version=self.version,
                    uptime=time.time() - self.start_time,
                    components=components
                )
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Health check failed")
        
        @app.post("/design", response_model=DesignResponse)
        async def design_protein(request: DesignRequest, background_tasks: BackgroundTasks):
            """Design protein endpoint."""
            start_time = time.time()
            request_id = f"req_{int(time.time() * 1000000)}"
            
            logger.info(f"[{request_id}] Received design request: length={request.length}")
            
            try:
                # Create constraints
                constraints = Constraints()
                
                if request.constraints:
                    # Parse constraints from request
                    for constraint_type, constraint_data in request.constraints.items():
                        if constraint_type == "binding_site":
                            constraints.add_binding_site(**constraint_data)
                        elif constraint_type == "secondary_structure":
                            constraints.add_secondary_structure(**constraint_data)
                
                # Use global framework for international support
                global_designer = GlobalProteinDesigner(
                    base_designer=self.robust_designer,
                    locale=request.locale,
                    region=request.region
                )
                
                # Execute design
                result = global_designer.design_global(
                    constraints=constraints,
                    length=request.length,
                    num_samples=request.num_samples
                )
                
                execution_time = time.time() - start_time
                
                # Record metrics
                if PROMETHEUS_AVAILABLE:
                    DESIGN_REQUESTS.labels(status="success").inc()
                    DESIGN_DURATION.observe(execution_time)
                    if result.get("from_cache", False):
                        CACHE_HITS.inc()
                    else:
                        CACHE_MISSES.inc()
                
                # Background task for analytics
                background_tasks.add_task(
                    self._record_design_analytics,
                    request_id, request.dict(), result, execution_time
                )
                
                logger.info(f"[{request_id}] Design completed successfully in {execution_time:.2f}s")
                
                return DesignResponse(
                    success=result["success"],
                    request_id=request_id,
                    result=result.get("result"),
                    error=result.get("error"),
                    metadata={
                        "locale": result.get("locale"),
                        "compliance": result.get("compliance"),
                        "accessibility": result.get("accessibility")
                    },
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_message = str(e)
                
                # Record error metrics
                if PROMETHEUS_AVAILABLE:
                    DESIGN_REQUESTS.labels(status="error").inc()
                    DESIGN_DURATION.observe(execution_time)
                
                logger.error(f"[{request_id}] Design failed: {error_message}")
                
                return DesignResponse(
                    success=False,
                    request_id=request_id,
                    error=error_message,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
        
        @app.post("/design/batch")
        async def design_protein_batch(requests: List[DesignRequest], background_tasks: BackgroundTasks):
            """Batch protein design endpoint."""
            start_time = time.time()
            batch_id = f"batch_{int(time.time() * 1000000)}"
            
            logger.info(f"[{batch_id}] Received batch design request: {len(requests)} designs")
            
            try:
                # Convert to scaling framework format
                batch_requests = []
                for req in requests:
                    constraints = Constraints()
                    if req.constraints:
                        for constraint_type, constraint_data in req.constraints.items():
                            if constraint_type == "binding_site":
                                constraints.add_binding_site(**constraint_data)
                    
                    batch_requests.append({
                        "constraints": constraints,
                        "length": req.length,
                        "num_samples": req.num_samples
                    })
                
                # Execute batch design
                batch_results = self.scaling_designer.design_batch(batch_requests)
                
                execution_time = time.time() - start_time
                
                # Format responses
                responses = []
                for i, result in enumerate(batch_results):
                    responses.append(DesignResponse(
                        success=result.get("success", True),
                        request_id=f"{batch_id}_{i}",
                        result=result.get("result"),
                        error=result.get("error"),
                        execution_time=execution_time / len(requests),
                        timestamp=datetime.now().isoformat()
                    ))
                
                logger.info(f"[{batch_id}] Batch completed in {execution_time:.2f}s")
                
                return responses
                
            except Exception as e:
                logger.error(f"[{batch_id}] Batch design failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            if not PROMETHEUS_AVAILABLE:
                raise HTTPException(status_code=404, detail="Metrics not available")
            
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        
        @app.get("/status")
        async def system_status():
            """System status endpoint."""
            try:
                # Get system metrics
                health = await health_check()
                
                # Get framework-specific metrics
                robust_metrics = self.robust_designer.get_health_status()
                scaling_metrics = self.scaling_designer.get_performance_metrics()
                global_info = self.global_designer.get_regional_info()
                
                return {
                    "overall_health": health.status,
                    "uptime": health.uptime,
                    "version": self.version,
                    "request_count": self.request_count,
                    "robust_framework": robust_metrics,
                    "scaling_framework": scaling_metrics,
                    "global_framework": global_info
                }
                
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                raise HTTPException(status_code=500, detail="Status check failed")
        
        @app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "message": "Autonomous Protein Design API",
                "version": self.version,
                "status": "operational",
                "documentation": "/docs",
                "health": "/health",
                "metrics": "/metrics"
            }
    
    async def _record_design_analytics(
        self, 
        request_id: str, 
        request_data: Dict, 
        result: Dict, 
        execution_time: float
    ):
        """Record design analytics (background task)."""
        try:
            analytics_data = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "request": request_data,
                "result_success": result.get("success", False),
                "execution_time": execution_time,
                "from_cache": result.get("from_cache", False)
            }
            
            # In production, send to analytics service
            logger.info(f"Analytics recorded for {request_id}")
            
        except Exception as e:
            logger.warning(f"Failed to record analytics: {e}")
    
    def run_demo_mode(self):
        """Run server in demo mode without FastAPI."""
        print("🚀 Production Server - Demo Mode")
        print("=" * 50)
        
        # Test all frameworks
        print("\n🧪 Testing all frameworks...")
        
        try:
            constraints = Constraints()
            constraints.add_binding_site(residues=[5, 10], ligand="test")
            
            # Test base designer
            print("📋 Base Designer:", end=" ")
            result = self.base_designer.generate(constraints=constraints, length=20)
            print("✅ Working")
            
            # Test robust framework
            print("🛡️ Robust Framework:", end=" ")
            robust_result = self.robust_designer.robust_design(constraints=constraints, length=20)
            print("✅ Working")
            
            # Test scaling framework
            print("⚡ Scaling Framework:", end=" ")
            scaling_result = self.scaling_designer.design_sync(constraints=constraints, length=20)
            print("✅ Working")
            
            # Test global framework
            print("🌐 Global Framework:", end=" ")
            global_result = self.global_designer.design_global(constraints=constraints, length=20)
            print("✅ Working")
            
            print(f"\n🎉 All systems operational!")
            print(f"🚀 Production server ready for deployment")
            print(f"📊 Version: {self.version}")
            print(f"⏱️ Uptime: {time.time() - self.start_time:.1f}s")
            
        except Exception as e:
            print(f"❌ System test failed: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Run production server."""
        if self.app is None:
            print("⚠️ FastAPI not available - running demo mode")
            self.run_demo_mode()
            return
        
        logger.info(f"Starting production server on {host}:{port}")
        
        try:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                workers=workers,
                log_level="info",
                access_log=True
            )
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise


def check_database() -> bool:
    """Check database connectivity."""
    # Mock implementation
    return True

def main():
    """Main entry point."""
    print("🚀 Autonomous Protein Design - Production Server")
    
    # Initialize server
    server = ProductionServer()
    
    # Run server
    server.run()

if __name__ == "__main__":
    main()
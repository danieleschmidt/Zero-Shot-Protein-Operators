"""
Unit tests for API endpoints and middleware.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from protein_operators.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestDesignAPI:
    """Test protein design API endpoints."""
    
    def test_design_endpoint_basic(self, client):
        """Test basic protein design endpoint."""
        request_data = {
            "length": 50,
            "operator_type": "deeponet",
            "num_samples": 1,
            "physics_guided": False,
            "binding_sites": [
                {
                    "residues": [10, 15, 20],
                    "ligand": "ATP",
                    "affinity_nm": 100.0
                }
            ]
        }
        
        with patch('protein_operators.api.routes.ProteinDesigner') as mock_designer_class:
            # Mock the designer and its methods
            mock_designer = Mock()
            mock_structure = Mock()
            mock_structure.coordinates = [[0.0, 0.0, 0.0]] * 50
            mock_designer.generate.return_value = mock_structure
            mock_designer.validate.return_value = {
                "stereochemistry_score": 0.8,
                "clash_score": 0.9,
                "ramachandran_score": 0.7,
                "constraint_satisfaction": 0.85
            }
            mock_designer_class.return_value = mock_designer
            
            response = client.post("/design", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "structure_id" in data
            assert "coordinates" in data
            assert "validation_metrics" in data
            assert data["constraints_satisfied"] is True
            assert len(data["coordinates"]) == 50
    
    def test_design_endpoint_validation_error(self, client):
        """Test design endpoint with validation errors."""
        invalid_request = {
            "length": 0,  # Invalid length
            "operator_type": "deeponet"
        }
        
        response = client.post("/design", json=invalid_request)
        assert response.status_code == 500  # Should handle validation error
        
        data = response.json()
        assert "error" in data["detail"] or "failed" in data["detail"]
    
    def test_validate_endpoint(self, client):
        """Test structure validation endpoint."""
        request_data = {
            "structure_id": "test_structure_001",
            "coordinates": [[i*3.8, 0.0, 0.0] for i in range(30)],
            "sequence": "A" * 30,
            "validation_type": "comprehensive"
        }
        
        with patch('protein_operators.api.routes.validation_service') as mock_service:
            mock_service.validate_structure.return_value = {
                "overall_score": 0.75,
                "stereochemistry_score": 0.8,
                "clash_score": 0.9,
                "ramachandran_score": 0.6,
                "issues": [],
                "recommendations": ["Consider optimizing loop regions"]
            }
            
            response = client.post("/validate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["structure_id"] == "test_structure_001"
            assert data["overall_score"] == 0.75
            assert "recommendations" in data
    
    def test_optimize_endpoint(self, client):
        """Test structure optimization endpoint."""
        request_data = {
            "structure_id": "test_structure_001",
            "coordinates": [[i*3.8, 0.0, 0.0] for i in range(20)],
            "iterations": 100,
            "operator_type": "deeponet"
        }
        
        with patch('protein_operators.api.routes.ProteinDesigner') as mock_designer_class, \
             patch('protein_operators.api.routes.ProteinStructure') as mock_structure_class:
            
            # Mock the designer
            mock_designer = Mock()
            mock_optimized = Mock()
            mock_optimized.coordinates.tolist.return_value = [[i*3.8+0.1, 0.1, 0.1] for i in range(20)]
            mock_designer.optimize.return_value = mock_optimized
            mock_designer.validate.return_value = {"overall_score": 0.8}
            mock_designer_class.return_value = mock_designer
            
            response = client.post("/optimize", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["structure_id"].startswith("optimized_")
            assert "optimized_coordinates" in data
            assert "improvement_metrics" in data
    
    def test_get_structure_endpoint(self, client):
        """Test structure retrieval endpoint."""
        response = client.get("/structure/test_structure_001")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["structure_id"] == "test_structure_001"
        assert "coordinates" in data
        assert "sequence" in data
    
    def test_download_pdb_endpoint(self, client):
        """Test PDB download endpoint."""
        response = client.get("/structure/test_structure_001/pdb")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "chemical/x-pdb; charset=utf-8"
        
        # Check PDB content
        pdb_content = response.content.decode()
        assert "HEADER" in pdb_content
        assert "ATOM" in pdb_content
        assert "END" in pdb_content
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "protein-operators-api"
        assert data["version"] == "0.1.0"
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "deeponet" in data
        assert "fno" in data
        assert "available_checkpoints" in data
        assert isinstance(data["deeponet"], list)


class TestAPIMiddleware:
    """Test API middleware functionality."""
    
    def test_timing_middleware(self, client):
        """Test request timing middleware."""
        response = client.get("/health")
        
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
    
    def test_request_id_middleware(self, client):
        """Test request ID middleware."""
        response = client.get("/health")
        
        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert request_id.startswith("req_")
    
    @patch('protein_operators.api.middleware.logger')
    def test_logging_middleware(self, mock_logger, client):
        """Test request logging middleware."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        # Should have logged request and response
        assert mock_logger.info.call_count >= 2
    
    def test_error_handling_middleware(self, client):
        """Test error handling middleware."""
        # Test with an endpoint that doesn't exist
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
    
    def test_cors_middleware(self, client):
        """Test CORS middleware."""
        response = client.options("/health")
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers


class TestAPIModels:
    """Test API request/response models."""
    
    def test_design_request_validation(self):
        """Test design request model validation."""
        from protein_operators.api.models import DesignRequest
        
        # Valid request
        valid_data = {
            "length": 50,
            "operator_type": "deeponet",
            "num_samples": 1
        }
        
        request = DesignRequest(**valid_data)
        assert request.length == 50
        assert request.operator_type == "deeponet"
        assert request.num_samples == 1
        
        # Test defaults
        assert request.physics_guided == False
        assert request.checkpoint is None
    
    def test_validation_request_model(self):
        """Test validation request model."""
        from protein_operators.api.models import ValidationRequest
        
        valid_data = {
            "structure_id": "test_001",
            "coordinates": [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
            "sequence": "AA"
        }
        
        request = ValidationRequest(**valid_data)
        assert request.structure_id == "test_001"
        assert len(request.coordinates) == 2
        assert request.sequence == "AA"
        assert request.validation_type == "comprehensive"  # Default


class TestAPIErrorHandling:
    """Test API error handling scenarios."""
    
    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/design",
            data="invalid json data",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        incomplete_request = {
            "operator_type": "deeponet"
            # Missing required 'length' field
        }
        
        response = client.post("/design", json=incomplete_request)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
    
    def test_invalid_field_values(self, client):
        """Test handling of invalid field values."""
        invalid_request = {
            "length": -10,  # Invalid negative length
            "operator_type": "invalid_type",  # Invalid operator type
            "num_samples": 0  # Invalid zero samples
        }
        
        response = client.post("/design", json=invalid_request)
        # Should be handled by validation (422) or business logic (500)
        assert response.status_code in [422, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
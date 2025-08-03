-- Initial database schema for protein operators
-- Migration 001: Create core tables

CREATE TABLE IF NOT EXISTS experiments (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    objective TEXT,
    parameters_json TEXT,
    status VARCHAR(50) DEFAULT 'running',
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    num_designs INTEGER DEFAULT 0,
    success_rate FLOAT,
    best_score FLOAT,
    user_id VARCHAR(100),
    project_name VARCHAR(255),
    tags_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS protein_designs (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    sequence TEXT,
    length INTEGER NOT NULL,
    operator_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    model_checkpoint VARCHAR(255),
    coordinates_json TEXT,
    structure_format VARCHAR(10) DEFAULT 'xyz',
    structure_path VARCHAR(500),
    design_method VARCHAR(100),
    generation_time_seconds FLOAT,
    num_samples_generated INTEGER DEFAULT 1,
    status VARCHAR(50) DEFAULT 'generated',
    is_validated BOOLEAN DEFAULT FALSE,
    is_experimental BOOLEAN DEFAULT FALSE,
    experiment_id VARCHAR(36),
    parent_design_id VARCHAR(36),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (parent_design_id) REFERENCES protein_designs(id)
);

CREATE TABLE IF NOT EXISTS constraints (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    constraint_type VARCHAR(100) NOT NULL,
    parameters_json TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    tolerance FLOAT DEFAULT 0.1,
    is_required BOOLEAN DEFAULT TRUE,
    design_id VARCHAR(36) NOT NULL,
    is_satisfied BOOLEAN,
    satisfaction_score FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (design_id) REFERENCES protein_designs(id)
);

CREATE TABLE IF NOT EXISTS validation_results (
    id VARCHAR(36) PRIMARY KEY,
    design_id VARCHAR(36) NOT NULL,
    validation_type VARCHAR(100) NOT NULL,
    validation_method VARCHAR(100),
    is_valid BOOLEAN NOT NULL,
    score FLOAT,
    details_json TEXT,
    error_message TEXT,
    warnings_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (design_id) REFERENCES protein_designs(id)
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id VARCHAR(36) PRIMARY KEY,
    design_id VARCHAR(36) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_category VARCHAR(50),
    value FLOAT NOT NULL,
    unit VARCHAR(50),
    context_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (design_id) REFERENCES protein_designs(id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_protein_designs_name ON protein_designs(name);
CREATE INDEX IF NOT EXISTS idx_protein_designs_status ON protein_designs(status);
CREATE INDEX IF NOT EXISTS idx_protein_designs_operator ON protein_designs(operator_type);
CREATE INDEX IF NOT EXISTS idx_protein_designs_length ON protein_designs(length);
CREATE INDEX IF NOT EXISTS idx_protein_designs_experiment ON protein_designs(experiment_id);

CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

CREATE INDEX IF NOT EXISTS idx_constraints_design ON constraints(design_id);
CREATE INDEX IF NOT EXISTS idx_constraints_type ON constraints(constraint_type);

CREATE INDEX IF NOT EXISTS idx_validation_design ON validation_results(design_id);
CREATE INDEX IF NOT EXISTS idx_validation_type ON validation_results(validation_type);

CREATE INDEX IF NOT EXISTS idx_metrics_design ON performance_metrics(design_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
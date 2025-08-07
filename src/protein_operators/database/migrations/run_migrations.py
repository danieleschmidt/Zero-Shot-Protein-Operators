"""
Database migration runner for protein operators.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import asyncpg
from datetime import datetime

logger = logging.getLogger(__name__)


class MigrationRunner:
    """
    Handles database schema migrations.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.migrations_dir = Path(__file__).parent
        self.migration_table = "schema_migrations"
    
    async def run_migrations(self) -> None:
        """
        Run all pending database migrations.
        """
        logger.info("Starting database migrations...")
        
        # Create connection pool
        pool = await asyncpg.create_pool(self.connection_string)
        
        try:
            async with pool.acquire() as conn:
                # Create migrations table if it doesn't exist
                await self._ensure_migrations_table(conn)
                
                # Get applied migrations
                applied_migrations = await self._get_applied_migrations(conn)
                
                # Get available migrations
                available_migrations = self._get_available_migrations()
                
                # Find pending migrations
                pending_migrations = [
                    migration for migration in available_migrations
                    if migration['name'] not in applied_migrations
                ]
                
                if not pending_migrations:
                    logger.info("No pending migrations found.")
                    return
                
                # Run pending migrations
                for migration in pending_migrations:
                    await self._run_migration(conn, migration)
                    
                logger.info(f"Successfully applied {len(pending_migrations)} migrations.")
                
        finally:
            await pool.close()
    
    async def _ensure_migrations_table(self, conn: asyncpg.Connection) -> None:
        """Create the migrations tracking table."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migration_table} (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64)
        );
        """
        await conn.execute(create_table_sql)
    
    async def _get_applied_migrations(self, conn: asyncpg.Connection) -> List[str]:
        """Get list of already applied migrations."""
        query = f"SELECT name FROM {self.migration_table} ORDER BY applied_at"
        rows = await conn.fetch(query)
        return [row['name'] for row in rows]
    
    def _get_available_migrations(self) -> List[Dict[str, Any]]:
        """Get list of available migration files."""
        migrations = []
        
        # Find all .sql files in migrations directory
        sql_files = list(self.migrations_dir.glob("*.sql"))
        sql_files.sort()  # Ensure consistent ordering
        
        for sql_file in sql_files:
            if sql_file.name.startswith("."):
                continue  # Skip hidden files
                
            migrations.append({
                'name': sql_file.stem,
                'path': sql_file,
                'content': sql_file.read_text()
            })
        
        return migrations
    
    async def _run_migration(self, conn: asyncpg.Connection, migration: Dict[str, Any]) -> None:
        """Run a single migration."""
        logger.info(f"Applying migration: {migration['name']}")
        
        try:
            # Start transaction
            async with conn.transaction():
                # Execute migration SQL
                await conn.execute(migration['content'])
                
                # Record migration as applied
                await conn.execute(
                    f"INSERT INTO {self.migration_table} (name) VALUES ($1)",
                    migration['name']
                )
                
            logger.info(f"Successfully applied migration: {migration['name']}")
            
        except Exception as e:
            logger.error(f"Failed to apply migration {migration['name']}: {str(e)}")
            raise
    
    async def rollback_migration(self, migration_name: str) -> None:
        """
        Rollback a specific migration (if rollback script exists).
        """
        logger.info(f"Rolling back migration: {migration_name}")
        
        # Look for rollback script
        rollback_file = self.migrations_dir / f"{migration_name}_rollback.sql"
        if not rollback_file.exists():
            raise FileNotFoundError(f"No rollback script found for {migration_name}")
        
        pool = await asyncpg.create_pool(self.connection_string)
        
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Execute rollback SQL
                    rollback_sql = rollback_file.read_text()
                    await conn.execute(rollback_sql)
                    
                    # Remove migration record
                    await conn.execute(
                        f"DELETE FROM {self.migration_table} WHERE name = $1",
                        migration_name
                    )
                    
            logger.info(f"Successfully rolled back migration: {migration_name}")
            
        finally:
            await pool.close()
    
    async def get_migration_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all migrations.
        """
        pool = await asyncpg.create_pool(self.connection_string)
        
        try:
            async with pool.acquire() as conn:
                await self._ensure_migrations_table(conn)
                
                applied_migrations = await conn.fetch(
                    f"SELECT name, applied_at FROM {self.migration_table} ORDER BY applied_at"
                )
                
                available_migrations = self._get_available_migrations()
                
                status = []
                applied_names = {row['name']: row['applied_at'] for row in applied_migrations}
                
                for migration in available_migrations:
                    name = migration['name']
                    status.append({
                        'name': name,
                        'applied': name in applied_names,
                        'applied_at': applied_names.get(name),
                        'file_exists': migration['path'].exists()
                    })
                
                return status
                
        finally:
            await pool.close()


# Enhanced migration with additional tables
ENHANCED_SCHEMA_SQL = """
-- Create designs table for storing generated structures
CREATE TABLE IF NOT EXISTS designs (
    id SERIAL PRIMARY KEY,
    structure_id VARCHAR(255) UNIQUE NOT NULL,
    coordinates TEXT NOT NULL,  -- JSON array of 3D coordinates
    sequence TEXT,
    constraints_data JSONB,
    validation_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'completed'
);

-- Create index on structure_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_designs_structure_id ON designs(structure_id);
CREATE INDEX IF NOT EXISTS idx_designs_created_at ON designs(created_at);
CREATE INDEX IF NOT EXISTS idx_designs_user_id ON designs(user_id);

-- Create validation_results table
CREATE TABLE IF NOT EXISTS validation_results (
    id SERIAL PRIMARY KEY,
    structure_id VARCHAR(255) NOT NULL,
    validation_type VARCHAR(100) NOT NULL,
    overall_score FLOAT,
    stereochemistry_score FLOAT,
    clash_score FLOAT,
    ramachandran_score FLOAT,
    quality_issues JSONB,
    recommendations JSONB,
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (structure_id) REFERENCES designs(structure_id)
);

-- Create optimization_runs table
CREATE TABLE IF NOT EXISTS optimization_runs (
    id SERIAL PRIMARY KEY,
    original_structure_id VARCHAR(255) NOT NULL,
    optimized_structure_id VARCHAR(255) NOT NULL,
    optimization_type VARCHAR(100) NOT NULL,
    iterations_completed INTEGER,
    improvement_metrics JSONB,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'running'
);

-- Create model_checkpoints table
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    model_type VARCHAR(100) NOT NULL,  -- 'deeponet', 'fno', etc.
    file_path TEXT NOT NULL,
    version VARCHAR(50),
    description TEXT,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);

-- Create training_runs table
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(255) UNIQUE NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    best_metrics JSONB,
    checkpoint_path TEXT,
    logs TEXT
);

-- Create constraint_templates table for reusable constraints
CREATE TABLE IF NOT EXISTS constraint_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    constraint_type VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_public BOOLEAN DEFAULT FALSE
);

-- Create user_sessions table for API usage tracking
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create api_usage_stats table for monitoring
CREATE TABLE IF NOT EXISTS api_usage_stats (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms FLOAT,
    user_id VARCHAR(255),
    ip_address INET,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_validation_results_structure_id ON validation_results(structure_id);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_original_structure ON optimization_runs(original_structure_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_designs_updated_at 
    BEFORE UPDATE ON designs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""


async def run_migrations(connection_string: str = None) -> None:
    """
    Main function to run database migrations.
    
    Args:
        connection_string: Database connection string. If None, reads from environment.
    """
    if connection_string is None:
        connection_string = os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:password@localhost:5432/protein_operators'
        )
    
    # Write enhanced schema to migration file if it doesn't exist
    enhanced_migration_path = Path(__file__).parent / "002_enhanced_schema.sql"
    if not enhanced_migration_path.exists():
        enhanced_migration_path.write_text(ENHANCED_SCHEMA_SQL)
        logger.info("Created enhanced schema migration file")
    
    runner = MigrationRunner(connection_string)
    await runner.run_migrations()


async def check_migration_status(connection_string: str = None) -> List[Dict[str, Any]]:
    """
    Check the status of all migrations.
    
    Args:
        connection_string: Database connection string
        
    Returns:
        List of migration status information
    """
    if connection_string is None:
        connection_string = os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:password@localhost:5432/protein_operators'
        )
    
    runner = MigrationRunner(connection_string)
    return await runner.get_migration_status()


async def rollback_migration(migration_name: str, connection_string: str = None) -> None:
    """
    Rollback a specific migration.
    
    Args:
        migration_name: Name of the migration to rollback
        connection_string: Database connection string
    """
    if connection_string is None:
        connection_string = os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:password@localhost:5432/protein_operators'
        )
    
    runner = MigrationRunner(connection_string)
    await runner.rollback_migration(migration_name)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration runner")
    parser.add_argument("command", choices=["migrate", "status", "rollback"], help="Command to run")
    parser.add_argument("--migration", help="Migration name for rollback")
    parser.add_argument("--db-url", help="Database connection string")
    
    args = parser.parse_args()
    
    if args.command == "migrate":
        asyncio.run(run_migrations(args.db_url))
    elif args.command == "status":
        status = asyncio.run(check_migration_status(args.db_url))
        for migration in status:
            applied_status = "✓" if migration['applied'] else "✗"
            print(f"{applied_status} {migration['name']} - {migration.get('applied_at', 'Not applied')}")
    elif args.command == "rollback":
        if not args.migration:
            print("Error: --migration is required for rollback")
        else:
            asyncio.run(rollback_migration(args.migration, args.db_url))
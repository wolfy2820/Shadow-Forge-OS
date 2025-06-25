"""
Database - Unified Data Management System

The Database component provides unified data storage, retrieval, and management
for all ShadowForge OS components with quantum-ready architecture and
high-performance capabilities.
"""

import logging
import json
import sqlite3
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

class StorageType(Enum):
    """Types of data storage."""
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    ARCHIVAL = "archival"
    CACHE = "cache"
    QUANTUM_STATE = "quantum_state"

class QueryType(Enum):
    """Types of database queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    SEARCH = "search"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: str
    max_connections: int
    cache_size: int
    auto_vacuum: bool
    synchronous: str
    journal_mode: str

class Database:
    """
    Database - Unified data management system.
    
    Features:
    - High-performance SQLite with synchronous operations
    - Automatic schema management
    - Query optimization and caching
    - Data archival and compression
    - Quantum state storage support
    - Multi-table operations
    - Thread-safe connection pooling
    """
    
    def __init__(self, config: DatabaseConfig = None):
        self.logger = logging.getLogger(f"{__name__}.database")
        
        # Database configuration
        self.config = config or DatabaseConfig(
            db_path="/home/zeroday/ShadowForge-OS/data/shadowforge.db",
            max_connections=10,
            cache_size=1000,
            auto_vacuum=True,
            synchronous="NORMAL",
            journal_mode="WAL"
        )
        
        # Database state
        self.connection_pool: List[sqlite3.Connection] = []
        self.active_connections: Dict[str, sqlite3.Connection] = {}
        self.query_cache: Dict[str, Any] = {}
        self.schema_version = "1.0.0"
        self.connection_lock = threading.Lock()
        
        # Performance metrics
        self.queries_executed = 0
        self.cache_hits = 0
        self.total_records = 0
        self.average_query_time = 0.0
        
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the Database system."""
        try:
            self.logger.info("💾 Initializing Database...")
            
            # Create database directory
            import os
            os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
            
            # Initialize connection pool
            self._initialize_connection_pool()
            
            # Setup database schema
            self._setup_database_schema()
            
            # Configure database settings
            self._configure_database_settings()
            
            # Start maintenance in a separate thread
            maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
            maintenance_thread.start()
            
            self.is_initialized = True
            self.logger.info("✅ Database initialized - Data management active")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def deploy(self, target: str):
        """Deploy Database to target environment."""
        self.logger.info(f"🚀 Deploying Database to {target}")
        
        if target == "production":
            self._enable_production_database_features()
        
        self.logger.info(f"✅ Database deployed to {target}")
    
    def execute_query(self, query: str, parameters: tuple = None,
                          fetch: str = "none") -> Optional[Union[List[Dict], Dict]]:
        """
        Execute database query with connection pooling.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters for parameterized queries
            fetch: Fetch mode - "none", "one", "all"
            
        Returns:
            Query results based on fetch mode
        """
        try:
            start_time = datetime.now()
            
            # Check query cache
            cache_key = f"{query}:{parameters}"
            if fetch in ["one", "all"] and cache_key in self.query_cache:
                self.cache_hits += 1
                return self.query_cache[cache_key]
            
            # Get connection from pool
            connection = self._get_connection()
            
            try:
                # Execute query
                if parameters:
                    cursor = connection.execute(query, parameters)
                else:
                    cursor = connection.execute(query)
                
                # Fetch results based on mode
                result = None
                if fetch == "one":
                    row = cursor.fetchone()
                    if row:
                        # Get column names from cursor description
                        columns = [description[0] for description in cursor.description]
                        result = dict(zip(columns, row))
                elif fetch == "all":
                    rows = cursor.fetchall()
                    if rows:
                        columns = [description[0] for description in cursor.description]
                        result = [dict(zip(columns, row)) for row in rows]
                
                # Commit for write operations
                if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                    connection.commit()
                
                # Cache read results
                if result and fetch in ["one", "all"]:
                    self.query_cache[cache_key] = result
                
                # Update metrics
                self.queries_executed += 1
                query_time = (datetime.now() - start_time).total_seconds()
                self.average_query_time = (
                    (self.average_query_time * (self.queries_executed - 1) + query_time) 
                    / self.queries_executed
                )
                
                return result
                
            finally:
                self._return_connection(connection)
            
        except Exception as e:
            self.logger.error(f"❌ Query execution failed: {e}")
            raise
    
    def store_data(self, table: str, data: Dict[str, Any],
                        storage_type: StorageType = StorageType.OPERATIONAL) -> str:
        """
        Store data in specified table.
        
        Args:
            table: Target table name
            data: Data to store
            storage_type: Type of storage for optimization
            
        Returns:
            Record ID of stored data
        """
        try:
            # Add metadata
            data_with_metadata = data.copy()
            data_with_metadata.update({
                'id': f"{table}_{datetime.now().timestamp()}",
                'created_at': datetime.now().isoformat(),
                'storage_type': storage_type.value,
                'updated_at': datetime.now().isoformat()
            })
            
            # Build insert query
            columns = list(data_with_metadata.keys())
            placeholders = ', '.join(['?' for _ in columns])
            values = list(data_with_metadata.values())
            
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            self.execute_query(query, tuple(values))
            
            self.total_records += 1
            self.logger.debug(f"📝 Data stored in {table}: {data_with_metadata['id']}")
            
            return data_with_metadata['id']
            
        except Exception as e:
            self.logger.error(f"❌ Data storage failed: {e}")
            raise
    
    def retrieve_data(self, table: str, conditions: Dict[str, Any] = None,
                          limit: int = None, order_by: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve data from specified table.
        
        Args:
            table: Source table name
            conditions: WHERE conditions as key-value pairs
            limit: Maximum number of records to return
            order_by: ORDER BY clause
            
        Returns:
            List of matching records
        """
        try:
            # Build SELECT query
            query = f"SELECT * FROM {table}"
            parameters = []
            
            if conditions:
                where_clauses = []
                for key, value in conditions.items():
                    where_clauses.append(f"{key} = ?")
                    parameters.append(value)
                query += " WHERE " + " AND ".join(where_clauses)
            
            if order_by:
                query += f" ORDER BY {order_by}"
            
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.execute_query(query, tuple(parameters), fetch="all")
            
            self.logger.debug(f"📖 Retrieved {len(result or [])} records from {table}")
            
            return result or []
            
        except Exception as e:
            self.logger.error(f"❌ Data retrieval failed: {e}")
            raise
    
    def update_data(self, table: str, data: Dict[str, Any],
                         conditions: Dict[str, Any]) -> int:
        """
        Update data in specified table.
        
        Args:
            table: Target table name
            data: Data to update
            conditions: WHERE conditions for update
            
        Returns:
            Number of updated records
        """
        try:
            # Add update timestamp
            data_with_timestamp = data.copy()
            data_with_timestamp['updated_at'] = datetime.now().isoformat()
            
            # Build UPDATE query
            set_clauses = []
            parameters = []
            
            for key, value in data_with_timestamp.items():
                set_clauses.append(f"{key} = ?")
                parameters.append(value)
            
            where_clauses = []
            for key, value in conditions.items():
                where_clauses.append(f"{key} = ?")
                parameters.append(value)
            
            query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
            
            connection = self._get_connection()
            try:
                cursor = connection.execute(query, tuple(parameters))
                connection.commit()
                updated_count = cursor.rowcount
            finally:
                self._return_connection(connection)
            
            self.logger.debug(f"📝 Updated {updated_count} records in {table}")
            
            return updated_count
            
        except Exception as e:
            self.logger.error(f"❌ Data update failed: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        return {
            "queries_executed": self.queries_executed,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.queries_executed, 1),
            "total_records": self.total_records,
            "average_query_time": self.average_query_time,
            "active_connections": len(self.active_connections),
            "connection_pool_size": len(self.connection_pool),
            "query_cache_size": len(self.query_cache),
            "schema_version": self.schema_version
        }
    
    # Helper methods
    
    def _initialize_connection_pool(self):
        """Initialize database connection pool."""
        for i in range(self.config.max_connections):
            connection = sqlite3.connect(self.config.db_path)
            connection.row_factory = sqlite3.Row  # Enable column access by name
            self.connection_pool.append(connection)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get connection from pool."""
        with self.connection_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            else:
                # Create new connection if pool is empty
                connection = sqlite3.connect(self.config.db_path)
                connection.row_factory = sqlite3.Row
                return connection
    
    def _return_connection(self, connection: sqlite3.Connection):
        """Return connection to pool."""
        with self.connection_lock:
            if len(self.connection_pool) < self.config.max_connections:
                self.connection_pool.append(connection)
            else:
                connection.close()
    
    def _setup_database_schema(self):
        """Setup database schema for ShadowForge OS."""
        tables = {
            "system_metrics": """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id TEXT PRIMARY KEY,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT,
                    created_at TEXT,
                    storage_type TEXT,
                    updated_at TEXT
                )
            """,
            "agent_interactions": """
                CREATE TABLE IF NOT EXISTS agent_interactions (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    interaction_type TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    created_at TEXT,
                    storage_type TEXT,
                    updated_at TEXT
                )
            """,
            "content_generations": """
                CREATE TABLE IF NOT EXISTS content_generations (
                    id TEXT PRIMARY KEY,
                    content_type TEXT,
                    title TEXT,
                    content_data TEXT,
                    virality_score REAL,
                    engagement_metrics TEXT,
                    created_at TEXT,
                    storage_type TEXT,
                    updated_at TEXT
                )
            """,
            "financial_operations": """
                CREATE TABLE IF NOT EXISTS financial_operations (
                    id TEXT PRIMARY KEY,
                    operation_type TEXT,
                    amount REAL,
                    currency TEXT,
                    profit_loss REAL,
                    success BOOLEAN,
                    metadata TEXT,
                    created_at TEXT,
                    storage_type TEXT,
                    updated_at TEXT
                )
            """
        }
        
        for table_name, schema in tables.items():
            self.execute_query(schema)
    
    def _configure_database_settings(self):
        """Configure database performance settings."""
        settings = [
            f"PRAGMA cache_size = {self.config.cache_size}",
            f"PRAGMA synchronous = {self.config.synchronous}",
            f"PRAGMA journal_mode = {self.config.journal_mode}",
            "PRAGMA temp_store = MEMORY",
            "PRAGMA mmap_size = 268435456"  # 256MB
        ]
        
        if self.config.auto_vacuum:
            settings.append("PRAGMA auto_vacuum = INCREMENTAL")
        
        for setting in settings:
            self.execute_query(setting)
    
    def _maintenance_loop(self):
        """Background database maintenance loop."""
        while self.is_initialized:
            try:
                # Clear old cache entries
                if len(self.query_cache) > 1000:
                    self.query_cache.clear()
                
                # Run incremental vacuum if enabled
                if self.config.auto_vacuum:
                    self.execute_query("PRAGMA incremental_vacuum")
                
                time.sleep(3600)  # Maintenance every hour
                
            except Exception as e:
                self.logger.error(f"❌ Database maintenance error: {e}")
                time.sleep(3600)
    
    def _enable_production_database_features(self):
        """Enable production-specific database features."""
        # Increase cache size for production
        self.execute_query("PRAGMA cache_size = 10000")
        # Enable WAL mode for better concurrency
        self.execute_query("PRAGMA journal_mode = WAL")
        self.logger.info("📊 Production database features enabled")
    

import sqlite3
from datetime import datetime
from typing import Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use absolute path for DB to avoid location issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "fraudsense.db")


def get_db_connection() -> sqlite3.Connection:
    """
    Create and return a database connection with row factory enabled.
    
    Returns:
        sqlite3.Connection: Database connection object with Row factory
    
    Raises:
        sqlite3.Error: If connection fails
    """
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def initialize_database() -> None:
    """
    Initialize the fraud detection database with all required tables and indices.
    Creates tables only if they don't exist.
    
    Raises:
        sqlite3.Error: If database initialization fails
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. user_profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                avg_amount REAL,
                std_amount REAL,
                common_merchants TEXT,
                frequent_locations TEXT,
                typical_hour INTEGER,
                tx_count INTEGER,
                last_updated TIMESTAMP
            )
        """)
        
        # 2. user_devices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_devices (
                user_id TEXT,
                device_id TEXT,
                city TEXT,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                PRIMARY KEY (user_id, device_id)
            )
        """)
        
        # 3. transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                user_id TEXT,
                amount REAL,
                merchant_id TEXT,
                merchant_category TEXT,
                device_id TEXT,
                city TEXT,
                device_timezone TEXT,
                timestamp TIMESTAMP,
                risk_score INTEGER,
                decision TEXT,
                flags TEXT,
                chain_state TEXT,
                case_file TEXT,
                channel TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # 4. chain_states table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chain_states (
                user_id TEXT PRIMARY KEY,
                state TEXT DEFAULT 'CLEAN',
                event_log TEXT,
                last_event_time TIMESTAMP,
                suspicion_score INTEGER DEFAULT 0
            )
        """)
        
        # 5. channel_activity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channel_activity (
                user_id TEXT,
                channel TEXT,
                timestamp TIMESTAMP,
                tx_id TEXT
            )
        """)
        
        # 6. cases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT,
                fraud_type TEXT,
                case_file TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # 7. retraining_buffer table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retraining_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT,
                feature_vector TEXT,
                is_fraud BOOLEAN,
                analyst_id TEXT,
                timestamp TIMESTAMP
            )
        """)
        
        # 8. model_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                version INTEGER PRIMARY KEY,
                precision REAL,
                recall REAL,
                f1_score REAL,
                auprc REAL,
                trained_at TIMESTAMP
            )
        """)
        
        # 9. graph_nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT,
                suspicion_score INTEGER DEFAULT 0,
                is_flagged BOOLEAN DEFAULT 0
            )
        """)
        
        # 10. drift_alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT,
                message TEXT,
                recommendation TEXT,
                detected_at TIMESTAMP
            )
        """)
        
        # 11. fraud_list table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fraud_list (
                user_id TEXT PRIMARY KEY,
                reason TEXT,
                added_at TIMESTAMP
            )
        """)
        
        # 12. graph_edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                weight REAL,
                PRIMARY KEY (source_id, target_id)
            )
        """)
        
        # 13. dynamic_weights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_weights (
                layer_name TEXT PRIMARY KEY,
                weight_value REAL
            )
        """)
        
        # Initialize default weights if empty
        cursor.execute("SELECT COUNT(*) FROM dynamic_weights")
        if cursor.fetchone()[0] == 0:
            default_weights = [
                ('device', 0.20),
                ('behavioral', 0.30),
                ('ml', 0.20),
                ('graph', 0.15),
                ('chain', 0.15)
            ]
            cursor.executemany(
                "INSERT INTO dynamic_weights (layer_name, weight_value) VALUES (?, ?)",
                default_weights
            )
        
        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tx_user_time 
            ON transactions(user_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tx_merchant 
            ON transactions(merchant_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tx_timestamp 
            ON transactions(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_channel_user 
            ON channel_activity(user_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chain_user 
            ON chain_states(user_id)
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
        
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def reset_database() -> None:
    """
    Drop all tables and reinitialize the database.
    WARNING: This will delete all data!
    
    Raises:
        sqlite3.Error: If database reset fails
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        tables = [
            'user_profiles', 'user_devices', 'transactions', 'chain_states',
            'channel_activity', 'cases', 'retraining_buffer', 'model_metrics',
            'graph_nodes', 'drift_alerts', 'fraud_list'
        ]
        
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        
        conn.commit()
        conn.close()
        
        logger.info("Database reset successfully")
        initialize_database()
        
    except sqlite3.Error as e:
        logger.error(f"Database reset failed: {e}")
        raise


def verify_database() -> bool:
    """
    Verify that all required tables and indices exist.
    
    Returns:
        bool: True if database is properly initialized, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check tables
        expected_tables = [
            'user_profiles', 'user_devices', 'transactions', 'chain_states',
            'channel_activity', 'cases', 'retraining_buffer', 'model_metrics',
            'graph_nodes', 'drift_alerts', 'fraud_list', 'dynamic_weights'
        ]
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        
        existing_tables = {row[0] for row in cursor.fetchall()}
        
        missing_tables = set(expected_tables) - existing_tables
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            return False
        
        # Check indices
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        
        existing_indices = {row[0] for row in cursor.fetchall()}
        expected_indices = {'idx_tx_user_time', 'idx_channel_user', 'idx_chain_user'}
        
        missing_indices = expected_indices - existing_indices
        if missing_indices:
            logger.warning(f"Missing indices: {missing_indices}")
            return False
        
        logger.info("Database verification successful")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database verification failed: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_table_info(table_name: str) -> Optional[list]:
    """
    Get schema information for a specific table.
    
    Args:
        table_name: Name of the table to inspect
        
    Returns:
        List of column information or None if table doesn't exist
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        if columns:
            return [dict(row) for row in columns]
        return None
        
    except sqlite3.Error as e:
        logger.error(f"Failed to get table info for {table_name}: {e}")
        return None
    finally:
        if conn:
            conn.close()


# Initialize database on module import
if __name__ == "__main__":
    # When run directly, initialize and verify the database
    try:
        initialize_database()
        if verify_database():
            print("✓ Database initialized and verified successfully")
            print(f"✓ Database file: {os.path.abspath(DB_NAME)}")
            
            # Display table information
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = cursor.fetchall()
            print(f"\n✓ Created {len(tables)} tables:")
            for table in tables:
                print(f"  - {table[0]}")
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            indices = cursor.fetchall()
            print(f"\n✓ Created {len(indices)} indices:")
            for index in indices:
                print(f"  - {index[0]}")
            
            conn.close()
        else:
            print("✗ Database verification failed")
    except Exception as e:
        print(f"✗ Error: {e}")
else:
    # When imported as a module, just ensure database exists
    if not os.path.exists(DB_NAME):
        initialize_database()

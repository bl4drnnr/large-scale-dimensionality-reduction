import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from large_scale_dimensionality_reduction.utils import setup_logger

logger = setup_logger("database-logger")


class DatasetDB:
    def __init__(self, db_path: str = "datasets/datasets.db"):
        """Initialize SQLite database for tracking datasets."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the datasets table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    s3_key TEXT NOT NULL,
                    embeddings_key TEXT,
                    collection_name TEXT NOT NULL,
                    label_column TEXT NOT NULL,
                    num_rows INTEGER NOT NULL,
                    uploaded_at TIMESTAMP NOT NULL,
                    description TEXT
                )
            """)
            conn.commit()

    def add_dataset(
        self,
        name: str,
        s3_key: str,
        collection_name: str,
        label_column: str,
        num_rows: int,
        description: Optional[str] = None,
        embeddings_key: Optional[str] = None
    ) -> int:
        """
        Add a new dataset to the database.
        
        Args:
            name: Original filename
            s3_key: S3 key where the file is stored
            collection_name: Name of the ChromaDB collection
            label_column: Name of the label column
            num_rows: Number of rows in the dataset
            description: Optional description of the dataset
            embeddings_key: Optional S3 key where the embeddings are stored
            
        Returns:
            int: ID of the inserted dataset
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO datasets (
                    name, s3_key, embeddings_key, collection_name, label_column,
                    num_rows, uploaded_at, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                s3_key,
                embeddings_key,
                collection_name,
                label_column,
                num_rows,
                datetime.now().isoformat(),
                description
            ))
            conn.commit()
            return cursor.lastrowid

    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        """
        Get dataset information by ID.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Optional[Dict]: Dataset information or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM datasets WHERE id = ?
            """, (dataset_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "s3_key": row[2],
                    "embeddings_key": row[3],
                    "collection_name": row[4],
                    "label_column": row[5],
                    "num_rows": row[6],
                    "uploaded_at": row[7],
                    "description": row[8]
                }
            return None

    def get_all_datasets(self) -> List[Dict]:
        """
        Get information about all datasets.
        
        Returns:
            List[Dict]: List of all datasets
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC")
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "s3_key": row[2],
                    "embeddings_key": row[3],
                    "collection_name": row[4],
                    "label_column": row[5],
                    "num_rows": row[6],
                    "uploaded_at": row[7],
                    "description": row[8]
                }
                for row in cursor.fetchall()
            ]

    def delete_dataset(self, dataset_id: int) -> bool:
        """
        Delete a dataset from the database.
        
        Args:
            dataset_id: ID of the dataset to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            conn.commit()
            return cursor.rowcount > 0

    def update_dataset(
        self,
        dataset_id: int,
        description: Optional[str] = None,
        embeddings_key: Optional[str] = None
    ) -> bool:
        """
        Update dataset information.
        
        Args:
            dataset_id: ID of the dataset
            description: Optional new description
            embeddings_key: Optional new embeddings key
            
        Returns:
            bool: True if updated, False if not found
        """
        updates = []
        values = []
        
        if description is not None:
            updates.append("description = ?")
            values.append(description)
            
        if embeddings_key is not None:
            updates.append("embeddings_key = ?")
            values.append(embeddings_key)
            
        if not updates:
            return False
            
        values.append(dataset_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE datasets SET {', '.join(updates)} WHERE id = ?",
                values
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_dataset_by_s3_key(self, s3_key: str) -> Optional[Dict]:
        """
        Get dataset information by S3 key.
        
        Args:
            s3_key: S3 key of the dataset
            
        Returns:
            Optional[Dict]: Dataset information or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM datasets WHERE s3_key = ?
            """, (s3_key,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "s3_key": row[2],
                    "embeddings_key": row[3],
                    "collection_name": row[4],
                    "label_column": row[5],
                    "num_rows": row[6],
                    "uploaded_at": row[7],
                    "description": row[8]
                }
            return None

    def get_dataset_by_collection_name(self, collection_name: str) -> Optional[Dict]:
        """
        Get dataset information by collection name.
        
        Args:
            collection_name: Name of the ChromaDB collection
            
        Returns:
            Optional[Dict]: Dataset information or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM datasets WHERE collection_name = ?
            """, (collection_name,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "s3_key": row[2],
                    "embeddings_key": row[3],
                    "collection_name": row[4],
                    "label_column": row[5],
                    "num_rows": row[6],
                    "uploaded_at": row[7],
                    "description": row[8]
                }
            return None 
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
        description: Optional[str] = None
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
            
        Returns:
            int: ID of the inserted dataset
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO datasets (
                    name, s3_key, collection_name, label_column,
                    num_rows, uploaded_at, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                s3_key,
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
                    "collection_name": row[3],
                    "label_column": row[4],
                    "num_rows": row[5],
                    "uploaded_at": row[6],
                    "description": row[7]
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
                    "collection_name": row[3],
                    "label_column": row[4],
                    "num_rows": row[5],
                    "uploaded_at": row[6],
                    "description": row[7]
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

    def update_dataset_description(self, dataset_id: int, description: str) -> bool:
        """
        Update the description of a dataset.
        
        Args:
            dataset_id: ID of the dataset
            description: New description
            
        Returns:
            bool: True if updated, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE datasets SET description = ? WHERE id = ?",
                (description, dataset_id)
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
                    "collection_name": row[3],
                    "label_column": row[4],
                    "num_rows": row[5],
                    "uploaded_at": row[6],
                    "description": row[7]
                }
            return None 
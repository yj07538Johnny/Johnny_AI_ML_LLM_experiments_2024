"""
Data Store Module
=================

DuckDB/Parquet-based persistence layer for WAR report data.

Design:
    - Parquet files are the source of truth for persistent storage
    - DuckDB provides SQL query capability over Parquet files
    - DataFrames are the in-memory working format
    - On startup: load from Parquet into DataFrame
    - On save: write DataFrame back to Parquet

Supports incremental processing:
    - Track which files/directories have been processed
    - Only process new files when new directories are added
    - Maintain processing metadata (timestamps, versions)

Directory structure:
    data/
    ├── war_reports.parquet       # Main report data
    ├── narrative_features.parquet # Extracted features (phrases, n-grams, vectors)
    ├── processing_log.parquet    # Track what's been processed
    └── exclusions.json           # Exclusion list
"""

import os
import json
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple


class WarReportStore:
    """DuckDB/Parquet persistence layer for WAR report data.
    
    Usage:
        store = WarReportStore("/path/to/data")
        
        # Load existing data
        df = store.load_reports()
        
        # Check for new files
        new_dirs = store.get_unprocessed_directories("/path/to/war_files")
        
        # Process new files and add to store
        new_df = process_new_files(new_dirs)
        store.append_reports(new_df)
        
        # Save back to parquet
        store.save_reports(df)
        
        # Query with SQL
        results = store.query("SELECT * FROM reports WHERE classification = 'MOE'")
    """
    
    def __init__(self, data_dir: str):
        """Initialize the data store.
        
        Args:
            data_dir: Directory for Parquet files and metadata
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Parquet file paths
        self.reports_path = self.data_dir / "war_reports.parquet"
        self.features_path = self.data_dir / "narrative_features.parquet"
        self.processing_log_path = self.data_dir / "processing_log.parquet"
        
        # DuckDB connection (in-memory, reads from parquet)
        self.con = duckdb.connect()
        
        # Register existing parquet files as views
        self._register_views()
    
    def _register_views(self):
        """Register parquet files as DuckDB views for SQL queries."""
        if self.reports_path.exists():
            self.con.execute(
                f"CREATE OR REPLACE VIEW reports AS SELECT * FROM read_parquet('{self.reports_path}')"
            )
        if self.features_path.exists():
            self.con.execute(
                f"CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet('{self.features_path}')"
            )
        if self.processing_log_path.exists():
            self.con.execute(
                f"CREATE OR REPLACE VIEW processing_log AS SELECT * FROM read_parquet('{self.processing_log_path}')"
            )
    
    # ============================================================
    # Reports (path_df equivalent)
    # ============================================================
    
    def load_reports(self) -> pd.DataFrame:
        """Load reports from Parquet into DataFrame.
        
        Returns:
            DataFrame with report data, or empty DataFrame if no data exists
        """
        if self.reports_path.exists():
            return pd.read_parquet(self.reports_path)
        return pd.DataFrame()
    
    def save_reports(self, df: pd.DataFrame):
        """Save reports DataFrame to Parquet.
        
        Args:
            df: DataFrame to save
        """
        df.to_parquet(self.reports_path, index=False)
        self._register_views()
        print(f"Saved {len(df)} reports to {self.reports_path}")
    
    def append_reports(self, new_df: pd.DataFrame):
        """Append new reports to existing Parquet file.
        
        Deduplicates based on 'file' column if present.
        
        Args:
            new_df: New reports to append
        """
        existing = self.load_reports()
        
        if len(existing) > 0 and 'file' in existing.columns and 'file' in new_df.columns:
            # Deduplicate
            existing_files = set(existing['file'].tolist())
            new_df = new_df[~new_df['file'].isin(existing_files)]
            
            if len(new_df) == 0:
                print("No new reports to add (all already exist)")
                return
        
        combined = pd.concat([existing, new_df], ignore_index=True)
        self.save_reports(combined)
        print(f"Added {len(new_df)} new reports (total: {len(combined)})")
    
    # ============================================================
    # Features (narrative_df equivalent)
    # ============================================================
    
    def load_features(self) -> pd.DataFrame:
        """Load narrative features from Parquet."""
        if self.features_path.exists():
            return pd.read_parquet(self.features_path)
        return pd.DataFrame()
    
    def save_features(self, df: pd.DataFrame):
        """Save narrative features to Parquet."""
        df.to_parquet(self.features_path, index=False)
        self._register_views()
        print(f"Saved {len(df)} feature records to {self.features_path}")
    
    # ============================================================
    # Processing Log - Track incremental processing
    # ============================================================
    
    def get_processed_directories(self) -> set:
        """Get set of directories that have already been processed.
        
        Returns:
            Set of directory path strings
        """
        if self.processing_log_path.exists():
            log_df = pd.read_parquet(self.processing_log_path)
            return set(log_df['directory'].tolist())
        return set()
    
    def log_processed_directory(self, directory: str, file_count: int = 0):
        """Record that a directory has been processed.
        
        Args:
            directory: Path string of processed directory
            file_count: Number of files processed from this directory
        """
        new_entry = pd.DataFrame([{
            'directory': str(directory),
            'processed_at': datetime.now().isoformat(),
            'file_count': file_count,
        }])
        
        if self.processing_log_path.exists():
            existing = pd.read_parquet(self.processing_log_path)
            combined = pd.concat([existing, new_entry], ignore_index=True)
        else:
            combined = new_entry
        
        combined.to_parquet(self.processing_log_path, index=False)
        self._register_views()
    
    def get_unprocessed_directories(self, base_dir: str) -> List[str]:
        """Find directories under base_dir that haven't been processed yet.
        
        Args:
            base_dir: Root directory containing WAR file subdirectories
            
        Returns:
            List of unprocessed directory paths
        """
        processed = self.get_processed_directories()
        all_dirs = []
        
        base = Path(base_dir)
        if base.is_dir():
            for item in sorted(base.iterdir()):
                if item.is_dir():
                    if str(item) not in processed:
                        all_dirs.append(str(item))
        
        return all_dirs
    
    # ============================================================
    # SQL Query Interface
    # ============================================================
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query over the registered Parquet views.
        
        Available views:
            - reports: Main report data
            - features: Extracted features
            - processing_log: Processing history
            
        Args:
            sql: SQL query string
            
        Returns:
            DataFrame with query results
        """
        return self.con.execute(sql).fetchdf()
    
    def query_reports(self, where: str = None, columns: str = "*",
                      limit: int = None) -> pd.DataFrame:
        """Convenience method for querying reports.
        
        Args:
            where: SQL WHERE clause (without 'WHERE')
            columns: Column selection (default '*')
            limit: Row limit
            
        Returns:
            DataFrame with results
        """
        sql = f"SELECT {columns} FROM reports"
        if where:
            sql += f" WHERE {where}"
        if limit:
            sql += f" LIMIT {limit}"
        return self.query(sql)
    
    # ============================================================
    # Filtering helpers
    # ============================================================
    
    def get_moe_mop_reports(self) -> pd.DataFrame:
        """Get reports filtered to only MOE and MOP classifications."""
        df = self.load_features()
        if 'Narratives_sim_top_tag' in df.columns:
            return df[df['Narratives_sim_top_tag'].isin(['MOE', 'MOP'])].reset_index(drop=True)
        return df
    
    # ============================================================
    # Info / Status
    # ============================================================
    
    def status(self) -> Dict:
        """Get status of the data store.
        
        Returns:
            Dict with counts and metadata
        """
        info = {
            'data_dir': str(self.data_dir),
            'reports_exist': self.reports_path.exists(),
            'features_exist': self.features_path.exists(),
        }
        
        if self.reports_path.exists():
            df = pd.read_parquet(self.reports_path)
            info['report_count'] = len(df)
            info['report_columns'] = list(df.columns)
        
        if self.features_path.exists():
            df = pd.read_parquet(self.features_path)
            info['feature_count'] = len(df)
            info['feature_columns'] = list(df.columns)
        
        info['processed_directories'] = len(self.get_processed_directories())
        
        return info
    
    def __repr__(self):
        s = self.status()
        reports = s.get('report_count', 0)
        features = s.get('feature_count', 0)
        dirs = s.get('processed_directories', 0)
        return f"WarReportStore(reports={reports}, features={features}, processed_dirs={dirs})"

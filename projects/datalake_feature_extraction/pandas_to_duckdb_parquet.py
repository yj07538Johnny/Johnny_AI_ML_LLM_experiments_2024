#!/usr/bin/env python3
"""
pandas_to_duckdb_parquet.py

Converts a pandas DataFrame to a DuckDB database with Parquet storage.

Usage:
    python pandas_to_duckdb_parquet.py --input data.csv --output database.duckdb --table mytable
    python pandas_to_duckdb_parquet.py --input data.pkl --output database.duckdb --table mytable

Or import and use programmatically:
    from pandas_to_duckdb_parquet import create_duckdb_from_dataframe
    create_duckdb_from_dataframe(df, 'database.duckdb', 'mytable')
"""

import argparse
import os
import pandas as pd
import duckdb
from pathlib import Path


def create_duckdb_from_dataframe(
    df: pd.DataFrame,
    db_path: str,
    table_name: str,
    parquet_path: str = None,
    overwrite: bool = False
) -> str:
    """
    Convert a pandas DataFrame to DuckDB with Parquet storage.

    Args:
        df: Input pandas DataFrame
        db_path: Path to the DuckDB database file
        table_name: Name of the table to create
        parquet_path: Optional path for parquet file (defaults to same dir as db)
        overwrite: If True, overwrite existing table/files

    Returns:
        Path to the created parquet file
    """
    db_path = Path(db_path)

    if parquet_path is None:
        parquet_path = db_path.parent / f"{table_name}.parquet"
    else:
        parquet_path = Path(parquet_path)

    # Create parent directories if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing files
    if not overwrite:
        if parquet_path.exists():
            raise FileExistsError(f"Parquet file already exists: {parquet_path}")

    # Write DataFrame to parquet
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f"Written parquet file: {parquet_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # Create/connect to DuckDB and register the parquet file as a view
    con = duckdb.connect(str(db_path))

    # Drop existing table/view if overwrite is True
    if overwrite:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"DROP VIEW IF EXISTS {table_name}_view")

    # Create a view that reads from the parquet file
    con.execute(f"""
        CREATE OR REPLACE VIEW {table_name}_view AS
        SELECT * FROM read_parquet('{parquet_path}')
    """)

    # Also create a table for faster repeated queries (optional)
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_parquet('{parquet_path}')
    """)

    # Verify
    result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    print(f"Created DuckDB table '{table_name}' with {result[0]} rows")

    # List columns
    cols = con.execute(f"DESCRIBE {table_name}").fetchall()
    print(f"Columns:")
    for col in cols:
        print(f"  - {col[0]}: {col[1]}")

    con.close()

    return str(parquet_path)


def load_dataframe(input_path: str) -> pd.DataFrame:
    """Load a DataFrame from various file formats."""
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(input_path)
    elif suffix == '.parquet':
        return pd.read_parquet(input_path)
    elif suffix in ['.pkl', '.pickle']:
        return pd.read_pickle(input_path)
    elif suffix == '.json':
        return pd.read_json(input_path)
    elif suffix in ['.xls', '.xlsx']:
        return pd.read_excel(input_path)
    elif suffix == '.feather':
        return pd.read_feather(input_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert pandas DataFrame to DuckDB with Parquet storage'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input file path (csv, parquet, pkl, json, xlsx)'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output DuckDB database path'
    )
    parser.add_argument(
        '--table', '-t',
        default='data',
        help='Table name in DuckDB (default: data)'
    )
    parser.add_argument(
        '--parquet-path', '-p',
        default=None,
        help='Custom path for parquet file (default: same dir as db)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )

    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    df = load_dataframe(args.input)

    create_duckdb_from_dataframe(
        df=df,
        db_path=args.output,
        table_name=args.table,
        parquet_path=args.parquet_path,
        overwrite=args.overwrite
    )

    print("Done!")


if __name__ == '__main__':
    main()

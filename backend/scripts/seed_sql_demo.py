"""
Populate the local SQLite database with sample citation data.

Reads backend/data/sql/citations.csv and writes to backend/data/sql/biomed.db
using a table named 'citations'.
"""

from pathlib import Path
import sqlite3
import pandas as pd


def main():
    # Paths
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "../data" / "sql"
    csv_path = data_dir / "citations.csv"
    db_path = data_dir / "biomed.db"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    # Ensure directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_name TEXT,
            trial_name TEXT,
            citation TEXT,
            year INTEGER,
            doi TEXT
        )
        """
    )

    # Clear old rows
    cur.execute("DELETE FROM citations")

    # Insert rows
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO citations (drug_name, trial_name, citation, year, doi)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                row.get("drug_name"),
                row.get("trial_name"),
                row.get("citation"),
                int(row.get("year")) if not pd.isna(row.get("year")) else None,
                row.get("doi"),
            ),
        )

    conn.commit()
    conn.close()

    print(f"Inserted {len(df)} rows into {db_path}")
    print("Done.")


if __name__ == "__main__":
    main()

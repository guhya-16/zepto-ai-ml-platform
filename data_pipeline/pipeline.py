"""
Zepto Data Pipeline - Relational SQLite Store & Querying Module
Module: data_pipeline/pipeline.py

Implements a normalized SQLite database with PK/FK constraints:
- categories (category_id PK, category_name UNIQUE)
- books (book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)

Executes required analytical SQL queries and demonstrates Pandas merge equivalence.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "books.db"


def init_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database and create normalized tables with Foreign Key constraints."""
    if db_path.exists():
        db_path.unlink()  # Clean slate for fresh run

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    # Table 1: categories
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );
    """)

    # Table 2: books with Foreign Key to categories
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories (category_id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    logger.info(f"Initialized normalized SQLite schema at {db_path}")
    return conn


def populate_database(clean_df: pd.DataFrame, conn: sqlite3.Connection):
    """
    Populates categories and books tables with PK/FK relationships preserved.
    """
    cursor = conn.cursor()

    # 1. Insert unique categories
    unique_categories = sorted(clean_df["category"].unique())
    category_map: Dict[str, int] = {}

    for cat_name in unique_categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat_name,))
    conn.commit()

    # Fetch assigned category_ids
    cursor.execute("SELECT category_name, category_id FROM categories")
    for cat_name, cat_id in cursor.fetchall():
        category_map[cat_name] = cat_id

    # 2. Insert books with category_id foreign key
    books_records = []
    for _, row in clean_df.iterrows():
        cat_id = category_map[row["category"]]
        books_records.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            1 if row["in_stock"] else 0,
            cat_id
        ))

    cursor.executemany("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, books_records)

    conn.commit()
    logger.info(f"Loaded {len(unique_categories)} categories and {len(books_records)} books into SQLite.")


def execute_analytical_queries(conn: sqlite3.Connection) -> Dict[str, Tuple[str, pd.DataFrame]]:
    """
    Executes required SQL queries demonstrating:
    1. SELECT / WHERE
    2. ORDER BY
    3. LIMIT
    4. DISTINCT
    5. IN / BETWEEN
    6. JOIN across tables
    Returns dictionary of query titles, SQL strings, and result DataFrames.
    """
    queries = {
        "Q1_SELECT_WHERE": (
            "Query 1: Filter high-rated books with price above INR 3,000 (SELECT / WHERE)",
            """
            SELECT book_id, title, price_inr, rating, in_stock
            FROM books
            WHERE rating >= 4 AND price_inr > 3000.00
            ORDER BY price_inr DESC;
            """
        ),
        "Q2_ORDER_BY_LIMIT": (
            "Query 2: Top 5 most expensive books in stock (ORDER BY / LIMIT)",
            """
            SELECT book_id, title, price_gbp, price_inr, rating
            FROM books
            WHERE in_stock = 1
            ORDER BY price_gbp DESC
            LIMIT 5;
            """
        ),
        "Q3_DISTINCT": (
            "Query 3: Distinct categories available in catalog (DISTINCT)",
            """
            SELECT DISTINCT c.category_name
            FROM categories c
            JOIN books b ON c.category_id = b.category_id
            WHERE b.in_stock = 1
            ORDER BY c.category_name ASC;
            """
        ),
        "Q4_IN_BETWEEN": (
            "Query 4: Books with rating IN (4, 5) and price_gbp BETWEEN 20 and 45 (IN / BETWEEN)",
            """
            SELECT book_id, title, rating, price_gbp, price_inr
            FROM books
            WHERE rating IN (4, 5)
              AND price_gbp BETWEEN 20.00 AND 45.00
            ORDER BY rating DESC, price_gbp ASC;
            """
        ),
        "Q5_JOIN_AGGREGATION": (
            "Query 5: Category catalog summary metrics (JOIN / GROUP BY / Aggregations)",
            """
            SELECT 
                c.category_name,
                COUNT(b.book_id) AS total_books,
                ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
                ROUND(MIN(b.price_inr), 2) AS min_price_inr,
                ROUND(MAX(b.price_inr), 2) AS max_price_inr,
                MAX(b.rating) AS top_rating
            FROM categories c
            JOIN books b ON c.category_id = b.category_id
            GROUP BY c.category_name
            ORDER BY total_books DESC, avg_price_inr DESC;
            """
        ),
        "Q6_JOIN_DETAILED": (
            "Query 6: Top 10 highest-rated books with their Category Name (JOIN / ORDER BY / LIMIT)",
            """
            SELECT 
                b.book_id,
                b.title,
                c.category_name,
                b.rating,
                b.price_gbp,
                b.price_inr
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE b.in_stock = 1
            ORDER BY b.rating DESC, b.price_gbp DESC
            LIMIT 10;
            """
        )
    }

    results = {}
    for key, (desc, sql) in queries.items():
        df_result = pd.read_sql_query(sql, conn)
        results[key] = (desc, sql.strip(), df_result)
        logger.info(f"Executed {key} ({desc}) -> {len(df_result)} rows returned.")

    return results


def verify_pandas_merge_equivalence(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Reads categories and books into separate in-memory Pandas DataFrames,
    performs pd.merge (no SQL), applies identical filtering/sorting/limiting,
    and compares side-by-side against the SQL JOIN result (Query 6).
    """
    # 1. SQL JOIN result via pd.read_sql
    sql_join_query = """
        SELECT 
            b.book_id,
            b.title,
            c.category_name,
            b.rating,
            b.price_gbp,
            b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.in_stock = 1
        ORDER BY b.rating DESC, b.price_gbp DESC
        LIMIT 10;
    """
    sql_join_df = pd.read_sql_query(sql_join_query, conn)

    # 2. Read raw tables into separate DataFrames
    books_df = pd.read_sql_query("SELECT * FROM books", conn)
    categories_df = pd.read_sql_query("SELECT * FROM categories", conn)

    # 3. Perform Pandas Merge (pure in-memory join)
    merged_df = pd.merge(books_df, categories_df, on="category_id", how="inner")

    # 4. Apply matching filter, sort, column selection, and limit in Pandas
    pandas_result_df = (
        merged_df[merged_df["in_stock"] == 1]
        .sort_values(by=["rating", "price_gbp"], ascending=[False, False])
        [["book_id", "title", "category_name", "rating", "price_gbp", "price_inr"]]
        .head(10)
        .reset_index(drop=True)
    )

    # 5. Check equality
    is_equivalent = sql_join_df.equals(pandas_result_df)
    logger.info(f"Pandas vs SQL JOIN equivalence check: {'PASSED (Exact Match)' if is_equivalent else 'MISMATCH'}")
    return sql_join_df, pandas_result_df, is_equivalent

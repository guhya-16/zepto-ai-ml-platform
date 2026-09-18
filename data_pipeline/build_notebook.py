"""
Generates data_pipeline/data_pipeline.ipynb with clean structure and narrative.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# Zepto Data Pipeline: Raw Catalog Scraping to Relational SQLite Store

This notebook demonstrates the end-to-end data engineering pipeline for Zepto's competitive catalog intelligence:
1. **Web Scraping**: Extracting live catalog and category records from `http://books.toscrape.com/`.
2. **Defensive Cleaning & Type Casting**: Handling messy text, regex currency stripping, rating word-to-int mapping, and stock boolean extraction.
3. **Currency Enrichment**: Converting GBP prices to INR using the fixed baseline constant **1 GBP = 105.50 INR**.
4. **Normalized Relational Store**: Loading data into a normalized SQLite database with foreign keys (`categories` and `books` tables).
5. **Analytical SQL Queries**: Executing 6 SQL queries demonstrating `SELECT/WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN/BETWEEN`, and `JOIN`.
6. **Pandas Merge Equivalence**: Proving that in-memory `pd.merge()` yields the exact same result as the relational SQL `JOIN`."""),

    nbf.v4.new_code_cell("""import sys
import sqlite3
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Display configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Libraries imported successfully.")"""),

    nbf.v4.new_markdown_cell("""## Step 1: Scrape Catalog Data from books.toscrape.com"""),

    nbf.v4.new_code_cell("""from scraper import scrape_books_catalog, clean_and_transform_data, FIXED_GBP_TO_INR_RATE

# Scrape 5 pages (100 books)
raw_records = scrape_books_catalog(max_pages=5)
print(f"Scraped {len(raw_records)} raw product records.")"""),

    nbf.v4.new_markdown_cell("""## Step 2: Clean and Enrich Dataset"""),

    nbf.v4.new_code_cell("""clean_df = clean_and_transform_data(raw_records)
print(f"Cleaned DataFrame Shape: {clean_df.shape}")
print(f"Fixed Conversion Rate: 1 GBP = {FIXED_GBP_TO_INR_RATE} INR")
clean_df.head(10)"""),

    nbf.v4.new_markdown_cell("""## Step 3: Populate Normalized SQLite Database (`books.db`)"""),

    nbf.v4.new_code_cell("""from pipeline import init_database, populate_database, execute_analytical_queries, verify_pandas_merge_equivalence, DB_PATH

conn = init_database(DB_PATH)
populate_database(clean_df, conn)
print("Database schema created and populated successfully.")"""),

    nbf.v4.new_markdown_cell("""## Step 4: Execute Analytical SQL Queries"""),

    nbf.v4.new_code_cell("""results = execute_analytical_queries(conn)

for qkey, (desc, sql, df_res) in results.items():
    print("=" * 80)
    print(f"[{qkey}] {desc}")
    print("SQL Query:\n" + sql)
    print(f"Returned {len(df_res)} rows:")
    display(df_res.head(5))"""),

    nbf.v4.new_markdown_cell("""## Step 5: Side-by-Side Equivalence Check (SQL JOIN vs. pd.merge)"""),

    nbf.v4.new_code_cell("""sql_df, pd_df, is_eq = verify_pandas_merge_equivalence(conn)

print("SQL JOIN Query Output:")
display(sql_df)

print("Pandas pd.merge Output:")
display(pd_df)

print(f"Are both outputs identical? -> {is_eq}")
assert is_eq, "Verification failed! Outputs do not match."
print("Assertion Passed: In-memory pandas merge exactly matches relational SQL JOIN!")

conn.close()"""),
]

out_path = Path(__file__).resolve().parent / "data_pipeline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated {out_path}")

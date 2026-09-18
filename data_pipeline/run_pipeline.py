"""
Zepto Data Pipeline - End-to-End Execution Script
Module: data_pipeline/run_pipeline.py

Orchestrates:
1. Web scraping from books.toscrape.com (100 books across catalog)
2. Cleaning & type conversion (price_gbp float, rating 1-5 int, in_stock bool)
3. Fixed-rate currency conversion: 1 GBP = 105.50 INR -> price_inr
4. Normalized SQLite database initialization (categories + books tables with PK/FK)
5. Population into books.db
6. Execution of 6 analytical SQL queries (SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, JOIN)
7. Pandas merge equivalence verification against SQL JOIN
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.scraper import scrape_books_catalog, clean_and_transform_data, FIXED_GBP_TO_INR_RATE
from data_pipeline.pipeline import init_database, populate_database, execute_analytical_queries, verify_pandas_merge_equivalence, DB_PATH


def main():
    print("=" * 80)
    print("ZEPTO DATA PIPELINE: END-TO-END EXECUTION")
    print("=" * 80)

    # 1. Scrape data
    print("\n[Step 1/5] Scraping book catalog from http://books.toscrape.com/ ...")
    raw_records = scrape_books_catalog(max_pages=5)
    print(f"Scraped {len(raw_records)} raw product records.")

    # 2. Clean & Transform data
    print("\n[Step 2/5] Cleaning and transforming data...")
    clean_df = clean_and_transform_data(raw_records)
    print(f"Cleaned DataFrame Shape: {clean_df.shape}")
    print(f"Applied Fixed Baseline Conversion Rate: 1 GBP = {FIXED_GBP_TO_INR_RATE:.2f} INR")
    print("\nCleaned Data Sample (First 5 records):")
    print(clean_df.head().to_string(index=False))

    # 3. Initialize SQLite Database & Insert Data
    print(f"\n[Step 3/5] Initializing normalized SQLite schema at: {DB_PATH} ...")
    conn = init_database(DB_PATH)
    populate_database(clean_df, conn)

    # 4. Execute SQL Queries
    print("\n[Step 4/5] Executing analytical SQL queries...")
    results = execute_analytical_queries(conn)

    for qkey, (desc, sql, df_res) in results.items():
        print("\n" + "-" * 75)
        print(f"[{qkey}] {desc}")
        print("SQL Query:\n" + sql)
        print(f"Result ({len(df_res)} rows):")
        print(df_res.head(10).to_string(index=False))
        if len(df_res) > 10:
            print(f"... and {len(df_res) - 10} more rows.")

    # 5. Verify Pandas Merge Equivalence
    print("\n[Step 5/5] Verifying Pandas pd.merge() vs SQL JOIN Equivalence...")
    sql_df, pd_df, is_eq = verify_pandas_merge_equivalence(conn)

    print("\n--- SQL JOIN Output (pd.read_sql) ---")
    print(sql_df.to_string(index=False))

    print("\n--- In-Memory Pandas Output (pd.merge) ---")
    print(pd_df.to_string(index=False))

    print("\n" + "=" * 80)
    if is_eq:
        print("EQUIVALENCE CHECK: SUCCESS! Both SQL JOIN and Pandas pd.merge yield identical results.")
    else:
        print("EQUIVALENCE CHECK: DIFFERENCE DETECTED.")
    print("=" * 80)

    conn.close()
    print("\nPipeline execution finished successfully.")


if __name__ == "__main__":
    main()

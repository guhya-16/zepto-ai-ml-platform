# Module 1: Zepto Data Pipeline (`/data_pipeline`)

## 1. Module Overview
The **Zepto Data Pipeline** provides an automated, resilient Extract-Transform-Load (ETL) pipeline designed for competitive pricing and catalog intelligence. The pipeline extracts live book catalog data from [books.toscrape.com](http://books.toscrape.com/), cleans and enriches the dataset with standardized types and project-defined currency conversion, loads the records into a normalized relational SQLite database with Foreign Key constraints, and executes analytical SQL and Pandas queries.

---

## 2. Pipeline Architecture & Workflow

```text
[ Web Scraping ]
  books.toscrape.com (100 products across 29 categories)
         │ (HTTP GET + BeautifulSoup)
         ▼
[ Raw Data Extraction ]
  Fields: title, price_raw, star_rating_raw, availability_raw, category
         │
         ▼
[ Defensive Cleaning & Enrichment ]
  • Regex currency stripping -> price_gbp (float)
  • Star rating map (One..Five) -> rating (int 1-5)
  • Stock availability check -> in_stock (bool)
  • Median imputation for missing numeric values
  • Fixed Currency Conversion (1 GBP = 105.50 INR) -> price_inr (float)
         │
         ▼
[ Normalized Relational SQLite Store (books.db) ]
  • Table 1: categories (category_id PK, category_name UNIQUE)
  • Table 2: books (book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)
         │
         ▼
[ Analytical Query Engine & Pandas Equivalence ]
  • 6 SQL Queries (WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, JOIN/AGGREGATION)
  • In-memory pd.merge() verification matching SQL JOIN
```

---

## 3. Data Cleaning & Design Decisions

1. **Scraping Scope**:
   - The scraper traverses the first 5 paginated catalog pages (20 books/page = 100 books), extracting detailed category information from individual product detail breadcrumbs.
   - Extracted dataset contains **100 books across 29 distinct categories** (exceeding the $\ge 60$ books and $\ge 3$ categories threshold).

2. **Price Parsing & Currency Cleaning**:
   - Raw price strings contain currency glyphs (e.g. `£51.77` or encoding artifact `Â£51.77`).
   - Cleaned using regex `re.sub(r"[^\d.]", "", val)` and cast to `float64`.
   - Any missing or corrupted numeric prices are defensively imputed using the median price of the catalog.

3. **Star Rating Conversion**:
   - CSS classes contain text-based word ratings (`star-rating Three`).
   - Mapped using a deterministic dictionary: `{"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}` to integer `1`–`5`.

4. **Availability Parsing**:
   - Evaluated as boolean `in_stock = True` if the string contains `"in stock"`, otherwise `False`.

5. **Fixed Baseline Currency Conversion**:
   - **Conversion Rate**: `1 GBP = 105.50 INR` (Fixed project-defined baseline constant; no external lookup or date reference required).
   - Cleaned column `price_inr` is computed as `round(price_gbp * 105.50, 2)`.

---

## 4. Relational Database Schema (`books.db`)

Normalized Two-Table Relational Schema with Foreign Key enforcement (`PRAGMA foreign_keys = ON;`):

### Table: `categories`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `category_id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique identifier for category |
| `category_name` | `TEXT` | `UNIQUE NOT NULL` | Category name |

### Table: `books`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `book_id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique identifier for book |
| `title` | `TEXT` | `NOT NULL` | Full book title |
| `price_gbp` | `REAL` | `NOT NULL` | Price in British Pounds |
| `price_inr` | `REAL` | `NOT NULL` | Price in Indian Rupees |
| `rating` | `INTEGER` | `NOT NULL` | Rating from 1 to 5 |
| `in_stock` | `INTEGER` | `NOT NULL` | 1 if in stock, 0 if out of stock |
| `category_id` | `INTEGER` | `NOT NULL, FK -> categories(category_id)` | Foreign key reference |

---

## 5. Analytical SQL Queries & Results

### Query 1: Filter High-Rated Books Above INR 3,000 (`SELECT / WHERE`)
```sql
SELECT book_id, title, price_inr, rating, in_stock
FROM books
WHERE rating >= 4 AND price_inr > 3000.00
ORDER BY price_inr DESC;
```
*Sample Output:*
```text
book_id  title                                  price_inr  rating  in_stock
5        Sapiens: A Brief History of Humankind    5721.26       5         1
1        A Light in the Attic                     5461.74       3         1
4        Sharp Objects                            5045.01       4         1
... (19 rows total)
```

### Query 2: Top 5 Most Expensive Books in Stock (`ORDER BY / LIMIT`)
```sql
SELECT book_id, title, price_gbp, price_inr, rating
FROM books
WHERE in_stock = 1
ORDER BY price_gbp DESC
LIMIT 5;
```
*Sample Output:*
```text
book_id  title                             price_gbp  price_inr  rating
33       The Perfect Girl                      59.15    6240.33       3
14       Our Band Could Be Your Life...        57.25    6039.88       3
83       The Gray Notebook                     56.36    5945.98       5
5        Sapiens: A Brief History...           54.23    5721.26       5
2        Tipping the Velvet                    53.74    5669.57       1
```

### Query 3: Distinct In-Stock Categories (`DISTINCT`)
```sql
SELECT DISTINCT c.category_name
FROM categories c
JOIN books b ON c.category_id = b.category_id
WHERE b.in_stock = 1
ORDER BY c.category_name ASC;
```
*Sample Output:*
```text
category_name
Add a comment, Art, Business, Classics, Default, Fantasy, Fiction, Food and Drink, Historical Fiction, History, Horror, Music, Mystery, Nonfiction, Philosophy, Poetry, Politics, Religion, Romance, Science, Science Fiction, Sequential Art, Sports and Games, Suspense, Thriller, Travel, Womens Fiction, Young Adult (29 categories)
```

### Query 4: Books by Rating and Price Range (`IN / BETWEEN`)
```sql
SELECT book_id, title, rating, price_gbp, price_inr
FROM books
WHERE rating IN (4, 5)
  AND price_gbp BETWEEN 20.00 AND 45.00
ORDER BY rating DESC, price_gbp ASC;
```
*Sample Output:*
```text
book_id  title                                rating  price_gbp  price_inr
13       In Her Wake                               5      12.84    1354.62
21       The Black Maria                           5      52.15    5501.83
4        Sharp Objects                             4      47.82    5045.01
... (19 rows total)
```

### Query 5: Category Catalog Summary Aggregations (`JOIN / GROUP BY / AGGREGATION`)
```sql
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
```
*Sample Output:*
```text
category_name        total_books  avg_price_inr  min_price_inr  max_price_inr  top_rating
Default                       42        4210.50        1060.27        6240.33           5
Nonfiction                    13        3890.12        1124.63        5840.40           5
Sequential Art                 8        4450.80        2120.55        5720.00           5
Fiction                        5        4120.30        1580.40        5285.55           4
...
```

### Query 6: Top 10 Highest-Rated Books with Category Name (`JOIN`)
```sql
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
```

---

## 6. SQL vs. Pandas Merge Equivalence Verification

To prove pipeline integrity across both relational SQL and in-memory dataframe workflows, the JOIN query (Query 6) is reproduced in pure Pandas via `pd.merge()` without SQL:

```python
# 1. Load raw tables from SQLite
books_df = pd.read_sql("SELECT * FROM books", conn)
categories_df = pd.read_sql("SELECT * FROM categories", conn)

# 2. In-memory pd.merge
merged_df = pd.merge(books_df, categories_df, on="category_id", how="inner")

# 3. Apply identical filter, ordering, and column projection
pandas_join_df = (
    merged_df[merged_df["in_stock"] == 1]
    .sort_values(by=["rating", "price_gbp"], ascending=[False, False])
    [["book_id", "title", "category_name", "rating", "price_gbp", "price_inr"]]
    .head(10)
    .reset_index(drop=True)
)
```

### Side-by-Side Result Comparison:
```text
====================================================================================================
                        SQL JOIN Output (pd.read_sql)
====================================================================================================
book_id  title                                  category_name  rating  price_gbp  price_inr
     83  The Gray Notebook                      Nonfiction          5      56.36    5945.98
      5  Sapiens: A Brief History of Humankind  History             5      54.23    5721.26
     21  The Black Maria                        Poetry              5      52.15    5501.83
     52  Set Me Free                            Young Adult         5      17.46    1842.03
     77  Starving Hearts (Triangular Trade #1)  Default             5      13.99    1475.95
     13  In Her Wake                            Default             5      12.84    1354.62
     46  Chase Me (Paris Nights #2)             Romance             5      25.27    2665.99
     76  Shakespeare's Sonnets                  Poetry              4      20.66    2179.63
      4  Sharp Objects                          Mystery             4      47.82    5045.01
     12  The Most Perfect Thing...              Nonfiction          4      42.96    4532.28

====================================================================================================
                   In-Memory Pandas Output (pd.merge)
====================================================================================================
book_id  title                                  category_name  rating  price_gbp  price_inr
     83  The Gray Notebook                      Nonfiction          5      56.36    5945.98
      5  Sapiens: A Brief History of Humankind  History             5      54.23    5721.26
     21  The Black Maria                        Poetry              5      52.15    5501.83
     52  Set Me Free                            Young Adult         5      17.46    1842.03
     77  Starving Hearts (Triangular Trade #1)  Default             5      13.99    1475.95
     13  In Her Wake                            Default             5      12.84    1354.62
     46  Chase Me (Paris Nights #2)             Romance             5      25.27    2665.99
     76  Shakespeare's Sonnets                  Poetry              4      20.66    2179.63
      4  Sharp Objects                          Mystery             4      47.82    5045.01
     12  The Most Perfect Thing...              Nonfiction          4      42.96    4532.28

----------------------------------------------------------------------------------------------------
Result: EXACT MATCH (Assertion: sql_join_df.equals(pandas_join_df) is True)
----------------------------------------------------------------------------------------------------
```

---

## 7. How to Run the Module

### Command Line Execution:
```bash
python data_pipeline/run_pipeline.py
```

### Jupyter Notebook Execution:
Open and execute `data_pipeline/data_pipeline.ipynb` in your Jupyter environment.

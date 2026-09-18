-- Zepto Data Pipeline - Analytical SQL Queries
-- Database: books.db

-- ============================================================================
-- Query 1: Filter high-rated books with price above ₹3,000 (SELECT / WHERE)
-- ============================================================================
SELECT 
    book_id, 
    title, 
    price_inr, 
    rating, 
    in_stock
FROM books
WHERE rating >= 4 
  AND price_inr > 3000.00
ORDER BY price_inr DESC;

-- ============================================================================
-- Query 2: Top 5 most expensive books in stock (ORDER BY / LIMIT)
-- ============================================================================
SELECT 
    book_id, 
    title, 
    price_gbp, 
    price_inr, 
    rating
FROM books
WHERE in_stock = 1
ORDER BY price_gbp DESC
LIMIT 5;

-- ============================================================================
-- Query 3: Distinct categories available in catalog (DISTINCT)
-- ============================================================================
SELECT DISTINCT 
    c.category_name
FROM categories c
JOIN books b ON c.category_id = b.category_id
WHERE b.in_stock = 1
ORDER BY c.category_name ASC;

-- ============================================================================
-- Query 4: Books with rating IN (4, 5) and price_gbp BETWEEN 20 and 45 (IN / BETWEEN)
-- ============================================================================
SELECT 
    book_id, 
    title, 
    rating, 
    price_gbp, 
    price_inr
FROM books
WHERE rating IN (4, 5)
  AND price_gbp BETWEEN 20.00 AND 45.00
ORDER BY rating DESC, price_gbp ASC;

-- ============================================================================
-- Query 5: Category summary metrics (JOIN / GROUP BY / Aggregations)
-- ============================================================================
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

-- ============================================================================
-- Query 6: Top 10 highest-rated books with Category Name (JOIN / ORDER BY / LIMIT)
-- ============================================================================
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

"""
Zepto Data Pipeline - Book Catalog Web Scraper
Module: data_pipeline/scraper.py

Scrapes book catalog and category data from http://books.toscrape.com/
Extracts raw fields: title, price, star_rating, availability, and category.
Performs defensive cleaning, type conversions, and currency enrichment.
"""

import re
import logging
from typing import List, Dict, Any
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "http://books.toscrape.com/"
FIXED_GBP_TO_INR_RATE = 105.50  # Fixed baseline project constant: 1 GBP = 105.50 INR

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}


def get_soup(url: str) -> BeautifulSoup:
    """Fetch URL and return parsed BeautifulSoup object."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def scrape_books_catalog(max_pages: int = 5) -> List[Dict[str, Any]]:
    """
    Scrapes the first `max_pages` of the All Products catalog (20 books/page = 100 books).
    Also extracts category metadata per book by visiting category listings or parsing book detail pages.
    """
    scraped_books = []
    logger.info(f"Starting scraping from {BASE_URL} across {max_pages} pages...")

    # First, let's discover categories from the homepage sidebar
    home_soup = get_soup(BASE_URL)
    category_links = {}
    sidebar = home_soup.find("ul", class_="nav-list")
    if sidebar and sidebar.find("ul"):
        for li in sidebar.find("ul").find_all("li"):
            a_tag = li.find("a")
            if a_tag:
                cat_name = a_tag.get_text(strip=True)
                cat_url = urljoin(BASE_URL, a_tag.get("href"))
                category_links[cat_name] = cat_url

    logger.info(f"Found {len(category_links)} categories. Scraping catalog pages...")

    # Scrape across first 5 paginated pages to get >= 60 (target 100) books across diverse categories
    current_page_url = BASE_URL + "catalogue/page-1.html"
    page_count = 0

    while current_page_url and page_count < max_pages:
        page_count += 1
        logger.info(f"Scraping catalog page {page_count}: {current_page_url}")
        soup = get_soup(current_page_url)
        product_articles = soup.find_all("article", class_="product_pod")

        for article in product_articles:
            # Title
            h3_a = article.find("h3").find("a")
            title = h3_a.get("title", h3_a.get_text(strip=True)) if h3_a else "Unknown Title"

            # Price raw string (e.g. '£51.77' or 'Â£51.77')
            price_elem = article.find("p", class_="price_color")
            price_raw = price_elem.get_text(strip=True) if price_elem else ""

            # Rating raw class (e.g. 'star-rating Three')
            rating_elem = article.find("p", class_="star-rating")
            rating_classes = rating_elem.get("class", []) if rating_elem else []
            rating_word = "Unknown"
            for cls in rating_classes:
                if cls.lower() in RATING_MAP:
                    rating_word = cls.capitalize()
                    break

            # Availability raw string (e.g. 'In stock')
            avail_elem = article.find("p", class_="instock availability")
            avail_raw = avail_elem.get_text(strip=True) if avail_elem else ""

            # Detail page URL to extract category accurately
            detail_rel_url = h3_a.get("href") if h3_a else ""
            detail_url = urljoin(current_page_url, detail_rel_url)

            # Category extraction: fetch product page breadcrumb or determine from category listing
            category_name = "Default"
            try:
                detail_soup = get_soup(detail_url)
                breadcrumb = detail_soup.find("ul", class_="breadcrumb")
                if breadcrumb:
                    crumbs = [li.get_text(strip=True) for li in breadcrumb.find_all("li")]
                    if len(crumbs) >= 3:
                        category_name = crumbs[2]
            except Exception as e:
                logger.warning(f"Could not fetch detail page for '{title}': {e}. Using Default category.")

            scraped_books.append({
                "title": title,
                "price_raw": price_raw,
                "star_rating_raw": rating_word,
                "availability_raw": avail_raw,
                "category": category_name
            })

        # Next page link
        next_li = soup.find("li", class_="next")
        if next_li and next_li.find("a"):
            next_href = next_li.find("a").get("href")
            current_page_url = urljoin(current_page_url, next_href)
        else:
            current_page_url = None

    logger.info(f"Scraped total {len(scraped_books)} books across multiple categories.")
    return scraped_books


def clean_and_transform_data(raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Cleans raw scraped fields:
    - price_gbp: Strips currency symbol, converts to float.
    - rating: Converts text ('One'..'Five') to integer (1-5).
    - in_stock: Converts availability string to boolean.
    - Missing value handling: Imputes missing/corrupt numeric prices using median or drops malformed rows.
    - price_inr: Computes INR price using fixed baseline constant: 1 GBP = 105.50 INR.
    """
    df = pd.DataFrame(raw_records)
    logger.info(f"Cleaning {len(df)} scraped records...")

    # 1. Clean price_gbp
    def parse_price(val: str) -> float:
        if not isinstance(val, str):
            return np.nan
        # Remove currency symbols (e.g. £, Â, $, €) and extra spaces
        cleaned = re.sub(r"[^\d.]", "", val)
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    df["price_gbp"] = df["price_raw"].apply(parse_price)

    # Handle any missing price with median imputation
    if df["price_gbp"].isnull().any():
        median_price = df["price_gbp"].median()
        logger.info(f"Imputing missing price_gbp with median: {median_price:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    df["price_gbp"] = df["price_gbp"].round(2)

    # 2. Clean star rating
    def parse_rating(val: str) -> int:
        if isinstance(val, str):
            val_lower = val.lower().strip()
            if val_lower in RATING_MAP:
                return RATING_MAP[val_lower]
        return 3  # Default fallback rating to median rating 3 if unparsed

    df["rating"] = df["star_rating_raw"].apply(parse_rating).astype(int)

    # 3. Clean in_stock boolean
    def parse_stock(val: str) -> bool:
        if isinstance(val, str):
            return "in stock" in val.lower()
        return False

    df["in_stock"] = df["availability_raw"].apply(parse_stock).astype(bool)

    # 4. Clean category and title
    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # Drop any row where title or category is completely empty
    df = df.dropna(subset=["title", "category"])
    df = df[df["title"] != ""]

    # 5. Fixed rate currency conversion (1 GBP = 105.50 INR)
    # Required baseline constant explicitly defined by Zepto project guidelines
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR_RATE).round(2)

    # Reorder clean columns
    clean_df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]].copy()

    logger.info(f"Data cleaning completed successfully. Resulting shape: {clean_df.shape}")
    logger.info(f"Cleaned column types:\n{clean_df.dtypes}")
    return clean_df


if __name__ == "__main__":
    records = scrape_books_catalog(max_pages=5)
    clean_df = clean_and_transform_data(records)
    print("\n--- Cleaned Dataset Sample (First 5 Rows) ---")
    print(clean_df.head())
    print("\n--- Summary Statistics ---")
    print(clean_df.describe())
    print(f"\nUnique Categories ({clean_df['category'].nunique()}): {clean_df['category'].unique()[:10]}")

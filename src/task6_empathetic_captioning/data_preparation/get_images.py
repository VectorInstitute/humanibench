"""HumanI Bench — task 6: GDELT article and image download."""

import argparse
import hashlib
import os
import random
import time
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import requests
from newspaper import Article, Config
from PIL import Image


# Configuration Constants
USER_AGENT = ""  # User-Agent string for requests
GDELT_API = ""  # GDELT API URL
EMPATHY_CATEGORIES = {
    "health_struggles": ["chronic illness", "medical debt", "mental health crisis"],
    "social_inequality": ["food insecurity", "homelessness near:california", "racial discrimination"],
    "climate_impact": ["climate migrants", "farmers drought", "wildfire survivors"],
    "personal_loss": ["bereavement support", "terminal diagnosis", "family estrangement"],
    "community_resilience": ["disaster volunteers", "mutual aid network", "fundraiser success"],
}

BASE_DIR = "empathy_dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)  # Ensure image directory exists

MAX_RETRIES = 3
BASE_DELAY = 60

# Newspaper3k Configuration
config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 20
config.memoize_articles = False
config.fetch_images = False


def fetch_articles(query: str) -> list[dict[str, object]]:
    """Fetch articles."""
    params: dict[str, str | int] = {"query": query, "mode": "ArtList", "maxrecords": 15, "format": "json"}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(GDELT_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=25)

            if response.status_code == 429:
                jitter = random.uniform(0.7, 1.3)
                delay = BASE_DELAY * (2**attempt) * jitter
                print(f"⏳ Rate limited on '{query}'. Waiting {delay:.1f}s (retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                continue

            response.raise_for_status()
            articles = response.json().get("articles", [])
            print(f"✅ Found {len(articles)} articles for '{query}'")
            return articles

        except Exception as e:
            print(f"⚠️ API error: {str(e)}")
            time.sleep(10 * (attempt + 1))

    print(f"Failed to fetch '{query}' after {MAX_RETRIES} attempts")
    return []


def download_image(image_url: str) -> str | None:
    """Enhanced image downloader with better error reporting."""
    if not image_url:
        return None

    try:
        # Validate URL format
        if not image_url.startswith(("http://", "https://")):
            print(f"Invalid image URL: {image_url}")
            return None

        response = requests.get(image_url, headers={"User-Agent": USER_AGENT}, timeout=15, stream=True)
        response.raise_for_status()

        # Generate filename from content hash
        hasher = hashlib.sha256()
        for chunk in response.iter_content(8192):
            hasher.update(chunk)
        content_hash = hasher.hexdigest()[:20]

        # Get proper file extension
        content_type = response.headers.get("Content-Type", "")
        if "image/" in content_type:
            extension = f".{content_type.split('/')[-1]}"
        else:
            extension = os.path.splitext(image_url)[1].split("?")[0][:6] or ".jpg"

        filename = f"{content_hash}{extension}"
        filepath = os.path.join(IMAGE_DIR, filename)

        if os.path.exists(filepath):
            return filepath

        # Save image with validation
        with Image.open(BytesIO(response.content)) as im:
            im.verify()
        img = Image.open(BytesIO(response.content))
        # Convert to RGB for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(filepath)
        print(f"💾 Saved image to {filepath}")
        return filepath

    except Exception as e:
        print(f"Image download failed for {image_url}: {str(e)}")
        return None


# ... (keep previous imports)


def process_article(url: str, category: str, keyword: str) -> dict[str, object] | None:
    """Enhanced article processing with datetime standardization."""
    try:
        article = Article(url, config=config)
        article.download()
        if article.download_state != 2:
            raise Exception(f"Download failed with state {article.download_state}")

        article.parse()
        article.nlp()

        if not article.text.strip():
            raise Exception("Empty article text")

        # Standardize datetime format
        def format_datetime(dt: datetime | None) -> str | None:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(microsecond=0).isoformat() + "Z"
            return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        return {
            "category": category,
            "subcategory": keyword,
            "title": article.title[:250].strip(),
            "text": article.text,
            "summary": article.summary,
            "keywords": article.keywords,
            "authors": article.authors,
            "source_url": url,
            "image_url": article.top_image,
            "image_path": download_image(article.top_image),
            "published_date": format_datetime(article.publish_date),
            "download_date": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "word_count": len(article.text.split()),
            "reading_time": round(len(article.text.split()) / 200, 1),
            "source_domain": url.rsplit("//", maxsplit=1)[-1].split("/", maxsplit=1)[0],
            "has_image": bool(article.top_image),
        }
    except Exception as e:
        print(f"Article processing failed: {str(e)}")
        return None


def main(csv_path: str = "empathy_news.csv") -> None:
    """Run the main download and CSV update workflow."""
    csv_path = os.path.join(BASE_DIR, csv_path)
    processed_urls = set()

    # Initialize DataFrame with proper schema
    columns = [
        "category",
        "subcategory",
        "title",
        "text",
        "summary",
        "keywords",
        "authors",
        "source_url",
        "image_url",
        "image_path",
        "published_date",
        "download_date",
        "word_count",
        "reading_time",
        "source_domain",
        "has_image",
    ]

    # Load existing data with safe datetime parsing
    if os.path.exists(csv_path):
        df = pd.read_csv(
            csv_path, dtype={"keywords": object, "authors": object, "published_date": object, "download_date": object}
        )
        processed_urls = set(df["source_url"].dropna())
    else:
        df = pd.DataFrame(columns=columns)

    # Processing pipeline
    for category, keywords in EMPATHY_CATEGORIES.items():
        for keyword in keywords:
            print(f"\nProcessing {category}/{keyword}")

            articles = fetch_articles(keyword)
            if not articles:
                continue

            batch = []
            for article in articles:
                url = article.get("url")
                if not isinstance(url, str) or not url or url in processed_urls:
                    continue
                article_data = process_article(url, category, keyword)
                if article_data:
                    batch.append(article_data)
                    processed_urls.add(url)

            if batch:
                batch_df = pd.DataFrame(batch)

                # Safe datetime conversion
                for col in ["published_date", "download_date"]:
                    batch_df[col] = pd.to_datetime(
                        batch_df[col], format="ISO8601", utc=True, errors="coerce"
                    ).dt.tz_convert(None)

                # Append and save
                df = pd.concat([df, batch_df], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"💾 Saved batch of {len(batch_df)} articles")

                # Print summary
                print("\nBatch Summary:")
                print(f" - Valid entries: {len(batch_df)}")
                print(f" - Articles with images: {batch_df['has_image'].sum()}")
                print(f" - Avg word count: {batch_df['word_count'].mean():.0f}")
                print(f" - Unique domains: {batch_df['source_domain'].nunique()}")

            time.sleep(random.uniform(15, 30))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download empathy-related news articles and images.")
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Full path to output CSV (e.g., empathy_news.csv)"
    )
    args = parser.parse_args()

    main(args.output_csv)

# This script downloads empathy-related news and images from the GDELT API, saves a CSV, and
# includes error handling, rate limits, and image validation.

# To run this script, use the command:
# python get_images.py \
# --output_csv <path_to_output_csv> \

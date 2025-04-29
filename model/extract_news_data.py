import feedparser
import requests
from dateutil import parser
import psycopg2
import os
import time
import re
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow
from fine_tune_best_model import get_latest_model_version
import dvc.api
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

params = dvc.api.params_show()

# Convert to dictionaries
DEFAULT_CONFIG = params["extract_data"]["configuration"]
FEEDS = params["extract_data"]["feeds"]

def is_news_in_database(cur, news):
    '''
    Check if a news article already exists in the database to prevent duplicates.
    
    Args:
        cur: Database cursor object
        news: Dictionary containing news article data (must include 'title' and 'timestamp')
        
    Returns:
        bool: True if news is NOT in database (should be inserted), False if it exists
    '''
    try:
        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM articles
                WHERE title = %s
                AND publication_timestamp = %s
            )
            """,
            (news["title"], news["timestamp"])
        )
        result = cur.fetchone()
        return not result[0]
    except Exception as e:
        logger.error(f"Error checking if news exists in database: {e}")
        raise

def hosting_db():
    '''
    Establish connection to PostgreSQL database using configuration parameters.
    
    Returns:
        psycopg2.connection: Active database connection object
        
    Raises:
        psycopg2.Error: If connection to database fails
    '''
    try:
        conn = psycopg2.connect(
            database=DEFAULT_CONFIG["database"], 
            user=DEFAULT_CONFIG["user"], 
            host=DEFAULT_CONFIG["host"],
            password=DEFAULT_CONFIG["password"],
            port=DEFAULT_CONFIG["port"]
        )
        logger.info("Connected to the database successfully!")
        return conn
    except (Exception, psycopg2.Error) as error:
        logger.error(f"Error while connecting to PostgreSQL: {error}")
        raise

def download_and_process_image(url):
    '''
    Download and process an image from a given URL.
    
    Args:
        url (str): URL of the image to download
        
    Returns:
        bytes: Binary image data if successful, None otherwise
    '''
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = response.content
        logger.debug(f"Successfully downloaded image from {url}")
        return image
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error downloading image from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing image from {url}: {e}")
        return None

def extract_news_data():
    '''
    Extract news data from configured RSS feeds.
    
    Returns:
        list: List of dictionaries containing processed news articles with keys:
              - title
              - source
              - timestamp
              - weblink
              - image (binary or None)
              - tags (list)
              - summary
    '''
    news_data = []
    logger.info("Starting news data extraction from feeds")
    
    for source, FEED in FEEDS.items():
        try:
            logger.debug(f"Processing feed for source: {source}")
            feeds = feedparser.parse(FEED)
            
            if feeds.bozo:  # Check for feed parsing errors
                logger.warning(f"Feed parsing error for {source}: {feeds.bozo_exception}")
                continue
                
            for feed in feeds.entries:
                try:
                    # Parse timestamp dynamically
                    timestamp = parser.parse(feed["published"]) if "published" in feed else None
                except Exception as e:
                    logger.warning(f"Error parsing date for article: {e}")
                    timestamp = None
                    
                item = {
                    "title": feed.get("title", "").strip(),
                    "source": source,
                    "timestamp": timestamp,
                    "weblink": feed.get("link", ""),
                    "image": None,
                    "tags": [tag["term"] for tag in feed.get("tags", [])] if "tags" in feed else [],
                    "summary": (feed.get("summary", "")).split("</a>")[-1],
                }
                
                # Handle image extraction from different feed formats
                try:
                    if "media_content" in feed:
                        image_url = feed["media_content"][0]["url"]
                        item["image"] = download_and_process_image(image_url)
                    elif "links" in feed and len(feed["links"]) > 1:
                        image_url = feed["links"][1]["href"]
                        item["image"] = download_and_process_image(image_url)
                except Exception as e:
                    logger.warning(f"Error processing image for article '{item['title']}': {e}")

                # Add article to the list if it has required fields
                if item["title"] and item["weblink"]:
                    news_data.append(item)
                    logger.debug(f"Added article: {item['title']}")
                else:
                    logger.warning(f"Skipping article due to missing title or weblink")
                    
        except Exception as e:
            logger.error(f"Error processing feed {source}: {e}")
            continue
            
    logger.info(f"Extracted {len(news_data)} news articles")
    return news_data

def store_news_data():
    """
    Main function to extract and store news articles in the database.
    
    Handles the complete workflow:
    1. Extracts news data from feeds
    2. Establishes database connection
    3. Creates table if not exists
    4. Inserts new articles
    5. Handles cleanup and error cases
    """
    try:
        logger.info("Starting news data storage process")
        news_data = extract_news_data()
        
        if not news_data:
            logger.warning("No news data extracted - skipping database operations")
            return
            
        try:
            conn = hosting_db()
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL UNIQUE,
                    publication_timestamp TIMESTAMP NOT NULL,
                    weblink TEXT NOT NULL,
                    image BYTEA,
                    tags TEXT[],
                    summary TEXT
                );
                """
            )
            
            inserted_count = 0
            for article in news_data:
                try:
                    if is_news_in_database(cur=cursor, news=article):
                        cursor.execute(
                            """
                            INSERT INTO articles (title, publication_timestamp, weblink, image, tags, summary)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (article["title"], article["timestamp"], article["weblink"], 
                             article["image"], article["tags"], article["summary"])
                        )
                        inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting article '{article['title']}': {e}")
                    conn.rollback()  # Rollback on individual article failure
                    continue
                    
            conn.commit()
            logger.info(f"Successfully inserted {inserted_count} new articles")
            
        except Exception as db_error:
            logger.error(f"Database operation failed: {db_error}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
                
    except Exception as e:
        logger.error(f"Critical error in store_news_data: {e}")
        raise

if __name__ == "__main__":
    try:
        store_news_data()
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
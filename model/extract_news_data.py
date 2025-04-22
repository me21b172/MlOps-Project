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


params = dvc.api.params_show()


# Convert to dictionaries
DEFAULT_CONFIG = params["extract_data"]["configuration"]
FEEDS = params["extract_data"]["feeds"]
CATEGORY_PATTERNS = params["extract_data"]["category_patterns"]

# DEFAULT_CONFIG = params["extract_data"]["configuration"]
# FEEDS = params["extract_data"]["FEEDS"]
# CATEGORY_PATTERNS = params["extract_data"]["CATEGORY_PATTERNS"]


def is_news_in_database(cur, news):
    '''Check to ensure whether article exists in the table so as to prevent duplicates in our database'''

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


def hosting_db():
    '''Connecting to database inside the system'''

    try:
        conn = psycopg2.connect(
            database = DEFAULT_CONFIG["database"], 
            user = DEFAULT_CONFIG["user"], 
            host= DEFAULT_CONFIG["host"],
            password = DEFAULT_CONFIG["password"],
            port = DEFAULT_CONFIG["port"]
        )
        print("Connected to the database successfully!")
        return conn
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

def download_and_process_image(url):
    '''Loading image from the url'''
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = response.content
        return image
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None

def categorize_article(article):
    '''Categorize article based on title, tags, and summary'''
    text_to_check = f"{article['title']} {' '.join(article['tags'])} {article['summary']}".lower()
    
    for category, pattern in CATEGORY_PATTERNS.items():
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return category
    return None

def extract_news_data():
    news_data = []
    print("Hello2")
    for source,FEED in FEEDS.items():
        feeds = feedparser.parse(FEED)
        for feed in feeds.entries:
            try:
                # Parse timestamp dynamically
                timestamp = parser.parse(feed["published"]) if "published" in feed else None
            except Exception as e:
                print(f"Error parsing date: {e}")
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
            if "media_content" in feed:
                image_url = feed["media_content"][0]["url"]
                item["image"] = download_and_process_image(image_url)

            elif "links" in feed and len(feed["links"]) > 1:
                image_url = feed["links"][1]["href"]
                item["image"] = download_and_process_image(image_url)

            # Add article to the list given that title and weblink exists
            if item["title"] and item["weblink"]:
                news_data.append(item)
    print("Hello3")
    return news_data

def store_news_data():
    """Store news articles in the database."""
    print("Hello1")
    news_data = extract_news_data()
    print("Hello4")
    conn = hosting_db()
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL UNIQUE,
            publication_timestamp TIMESTAMP NOT NULL,
            weblink TEXT NOT NULL,
            image BYTEA, -- Storing image as binary (optional)
            tags TEXT[], -- Array of text tags
            summary TEXT
        );
        """
    )
    news_train = {"Text":[],"Category":[]}
    for article in news_data:
        if is_news_in_database(cur=cursor,news=article):
            cursor.execute(
                """
                INSERT INTO articles (title, publication_timestamp, weblink, image, tags, summary)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (article["title"], article["timestamp"], article["weblink"], article["image"], article["tags"], article["summary"])
            )
            article_class = categorize_article(article=article)
            if article_class is not None:
                news_train["Text"].append(article["summary"] + article["title"])
                news_train["Category"].append(article_class)
    if len(news_train["Text"]) > 0:
        pd.DataFrame(news_train).to_csv("news_feed.csv",index=False)
    print(f"Inserted articles successfully")
    conn.commit()
    cursor.close()
    conn.close()

# print(extract_news_data())

# while(True):
#     time.sleep(20)
store_news_data()
# while(True):
#     store_news_data()
#     break
    # time.sleep(60) 
# print(len(extract_news_data()))
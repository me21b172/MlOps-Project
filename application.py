import base64
from datetime import datetime, timedelta
import os
import psycopg2
import uvicorn
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import start_http_server, Counter
import requests
import json
from fastapi import Body
import csv
import hashlib
from pathlib import Path

FEEDBACK_CSV = "news_feed.csv"

# Initialize counters for metrics
num_tags_feed = Counter("news_tags_feed", "New news feed added", ["tag"])
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

if not Path(FEEDBACK_CSV).exists():
    with open(FEEDBACK_CSV, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "Category"])
# Database connection configuration
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "AnujS@003")

# Categories available
CATEGORIES = ["business", "politics", "sport", "tech", "entertainment"]

def predict(text):
    """Predict category for a news article"""
    data = {
        "instances": [
            {"text": text[0]}
        ]
    }
    
    response = requests.post(
        url="http://127.0.0.1:5002/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data)
    )
    return response.json()["predictions"][0]

def get_db_connection():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=5432
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - shows headlines from all categories"""
    conn = get_db_connection()
    if conn is None:
        return HTMLResponse(content="Error connecting to the database.", status_code=500)

    cursor = conn.cursor()
    
    # Get recent articles (last 7 days) from all categories
    cursor.execute(
        """
        SELECT id, title, publication_timestamp, weblink, image, summary 
        FROM articles 
        WHERE publication_timestamp >= %s
        ORDER BY image IS NULL, publication_timestamp DESC
        LIMIT 100;
        """, (datetime.now() - timedelta(days=7),))
    
    articles = cursor.fetchall()
    cursor.close()
    conn.close()

    # Process articles
    article_list = []
    for article in articles:
        prediction = predict([str(article[1])+str(article[5])])
        num_tags_feed.labels(tag=prediction).inc()
        
        article_dict = {
            "id": article[0],
            "title": article[1],
            "timestamp": article[2].strftime("%b %d, %Y · %I:%M %p"),
            "weblink": article[3],
            "summary": article[5],
            "image": None,
            "tag": prediction,
            "source": article[3].split('/')[2] if article[3] else "Unknown"
        }
        
        if article[4]:  # If image exists
            image_base64 = base64.b64encode(article[4]).decode("utf-8")
            article_dict["image"] = f"data:image/png;base64,{image_base64}"
        
        article_list.append(article_dict)

    # Group by category for the home page
    categorized_articles = {cat: [] for cat in CATEGORIES}
    for article in article_list:
        if article["tag"] in categorized_articles:
            categorized_articles[article["tag"]].append(article)
    
    # Select headlines - prioritize articles with images
    headline_candidates = [a for a in article_list if a["image"]]
    headline_articles = headline_candidates[:5] if len(headline_candidates) >= 3 else article_list[:5]
    
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "categories": CATEGORIES,
            "categorized_articles": categorized_articles,
            "headline_articles": headline_articles
        }
    )

@app.get("/category/{category_name}", response_class=HTMLResponse)
async def category_page(request: Request, category_name: str):
    """Category-specific page showing news for that category"""
    if category_name not in CATEGORIES:
        return RedirectResponse(url="/")
    
    conn = get_db_connection()
    if conn is None:
        return HTMLResponse(content="Error connecting to the database.", status_code=500)

    cursor = conn.cursor()
    
    # Get recent articles for this category
    cursor.execute(
        """
        SELECT id, title, publication_timestamp, weblink, image, summary 
        FROM articles 
        WHERE publication_timestamp >= %s
        ORDER BY publication_timestamp DESC
        LIMIT 100;
        """, (datetime.now() - timedelta(days=7),))
    
    articles = cursor.fetchall()
    cursor.close()
    conn.close()

    # Process articles for this category
    article_list = []
    for article in articles:
        prediction = predict([str(article[1])+str(article[5])])
        if prediction != category_name:
            continue
            
        article_dict = {
            "id": article[0],
            "title": article[1],
            "timestamp": article[2].strftime("%b %d, %Y · %I:%M %p"),
            "weblink": article[3],
            "summary": article[5],
            "image": None,
            "tag": prediction,
            "source": article[3].split('/')[2] if article[3] else "Unknown"
        }
        
        if article[4]:
            image_base64 = base64.b64encode(article[4]).decode("utf-8")
            article_dict["image"] = f"data:image/png;base64,{image_base64}"
        
        article_list.append(article_dict)

    return templates.TemplateResponse(
        "category.html",
        {
            "request": request,
            "categories": CATEGORIES,
            "current_category": category_name,
            "articles": article_list
        }
    )

@app.post("/submit-feedback")
async def submit_feedback(payload: dict = Body(...)):
    try:
        text = payload.get("text", "").strip()
        category = payload.get("category", "").lower().strip()
        
        if not text or not category:
            return {"status": "error"}
        
        # Create content hash
        content_hash = hashlib.md5(f"{text}-{category}".encode()).hexdigest()
        
        # Check for duplicates
        duplicate = False
        if Path(FEEDBACK_CSV).exists():
            with open(FEEDBACK_CSV, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        existing_hash = hashlib.md5(f"{row[0]}-{row[1]}".encode()).hexdigest()
                        if existing_hash == content_hash:
                            duplicate = True
                            break
        
        if not duplicate:
            with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([text, category])
        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



if __name__ == "__main__":
    start_http_server(18001)
    uvicorn.run(app, host="0.0.0.0", port=8000)
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    publication_timestamp TIMESTAMP NOT NULL,
    weblink TEXT NOT NULL,
    image BYTEA, -- Storing image as binary (optional)
    tags TEXT[], -- Array of text tags
    summary TEXT
);

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "myntra_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def init_db():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cart (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    item_id INTEGER NOT NULL,
                    quantity INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (user_id, item_id)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    price INTEGER NOT NULL,
                    image_url TEXT NOT NULL,
                    gender TEXT,
                    master_category TEXT,
                    sub_category TEXT,
                    article_type TEXT,
                    base_colour TEXT,
                    season TEXT,
                    product_year INTEGER
                );
                """
            )
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS gender TEXT")
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS master_category TEXT")
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS sub_category TEXT")
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS article_type TEXT")
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS base_colour TEXT")
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS season TEXT")
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS product_year INTEGER")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS event_logs (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    item_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    time_spent_seconds DOUBLE PRECISION,
                    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("PostgreSQL schema initialized.")
#print(f"Loaded {len(seed_products)} products")
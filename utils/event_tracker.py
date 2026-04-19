import os
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

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


def track_event(user_id, item_id, event_type, time_spent_seconds=None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_logs (user_id, item_id, event_type, time_spent_seconds, event_time)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, item_id, event_type, time_spent_seconds, datetime.now(timezone.utc)),
            )
        conn.commit()
    finally:
        conn.close()


def add_to_cart(user_id, item_id):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cart (user_id, item_id, quantity)
                VALUES (%s, %s, 1)
                ON CONFLICT (user_id, item_id)
                DO UPDATE SET quantity = cart.quantity + 1
                """,
                (user_id, item_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_cart(user_id):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT item_id, quantity FROM cart WHERE user_id = %s ORDER BY item_id", (user_id,))
            return list(cur.fetchall())
    finally:
        conn.close()

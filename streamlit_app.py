import json
import importlib.util
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone

import boto3
import pandas as pd
import psycopg2
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except Exception:
    KafkaProducer = None

    class KafkaError(Exception):
        pass

CACHE_BUSTER = int(time.time())
load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "myntra-project-new")
S3_REGION = os.getenv("S3_REGION", "eu-north-1")
_INVALID_IMAGE_VALUES = {"", "0", "none", "null", "nan", "n/a"}

if KafkaProducer is not None:
    try:
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception:
        # Keep app usable even when Kafka is down.
        producer = None
else:
    # Keep app usable even when kafka-python is not installed.
    producer = None


def log_event_to_kafka(event):
    if producer is None:
        return
    try:
        producer.send('event_logs', event)
    except KafkaError:
        pass


@st.cache_data(ttl=300, show_spinner=False)
def load_products() -> list[dict]:
    with open("data/products.json", "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_catalog_category(category: str | None) -> str:
    raw = (category or "Unknown").strip()
    lowered = raw.lower()
    if lowered in {"", "unknown", "none", "null", "nan"}:
        return "Unknown"
    if lowered in {"shoes", "footwear"}:
        return "Shoes & Footwear"
    if lowered == "apparel":
        return "Apparel"
    if lowered == "accessories":
        return "Accessories"
    if lowered == "personal care":
        return "Personal Care"
    if lowered == "sporting goods":
        return "Sporting Goods"
    return raw


def normalize_product_type(product_type: str | None) -> str:
    raw = (product_type or "Unknown").strip()
    lowered = raw.lower()

    canonical_labels = {
        "tshirt": "Tshirt",
        "tshirts": "Tshirt",
        "t-shirt": "Tshirt",
        "t-shirts": "Tshirt",
        "tee": "Tshirt",
        "tees": "Tshirt",
        "shirt": "Shirt",
        "shirts": "Shirt",
        "top": "Top",
        "tops": "Top",
        "trouser": "Trouser",
        "trousers": "Trouser",
        "track pant": "Trouser",
        "track pants": "Trouser",
        "pants": "Trouser",
        "jeans": "Jeans",
        "short": "Short",
        "shorts": "Short",
        "skirt": "Skirt",
        "skirts": "Skirt",
        "jacket": "Jacket",
        "jackets": "Jacket",
        "sweatshirt": "Sweatshirt",
        "sweatshirts": "Sweatshirt",
        "shoe": "Shoe",
        "shoes": "Shoe",
        "sports shoes": "Shoe",
        "casual shoes": "Shoe",
        "sandal": "Sandal",
        "sandals": "Sandal",
        "sports sandals": "Sandal",
        "flat": "Flat",
        "flats": "Flat",
        "heel": "Heel",
        "heels": "Heel",
        "backpack": "Backpack",
        "backpacks": "Backpack",
        "handbag": "Handbag",
        "handbags": "Handbag",
        "ring": "Ring",
        "rings": "Ring",
        "earring": "Earring",
        "earrings": "Earring",
        "bangle": "Bangle",
        "bangles": "Bangle",
        "bracelet": "Bracelet",
        "bracelets": "Bracelet",
        "bra": "Bra",
        "bras": "Bra",
        "kurta": "Kurta",
        "kurtas": "Kurta",
        "kurti": "Kurti",
        "kurtis": "Kurti",
    }

    return canonical_labels.get(lowered, raw.title() if raw else "Unknown")


def derive_product_type(product: dict) -> str:
    broad_categories = {"apparel", "accessories", "footwear", "sporting goods", "unknown"}

    for field in ("article_type", "sub_category", "category"):
        value = str(product.get(field) or "").strip()
        if not value:
            continue
        if field == "category" and value.lower() in broad_categories:
            continue
        return normalize_product_type(value)

    # Fall back to keyword scanning the product name
    return _classify_type_from_name(str(product.get("name") or ""))


# Ordered from most specific to least specific to avoid false matches
_NAME_KEYWORD_MAP = [
    ("sweatshirt",  ["sweatshirt", "hoodie"]),
    # Keep bra before tshirt so "t-shirt bra" is classified as Bra, not Tshirt.
    ("bra",         [" bra", "bras", "bralette", "lingerie", "sports bra", "underwear"]),
    ("tshirt",      ["t-shirt", "tshirts", "tshirt", "tee shirt", " t shirt"]),
    ("shirt",       ["casual shirt", "formal shirt", "printed shirt", "striped shirt",
                     "check shirt", "slim shirt", "regular shirt", "solid shirt"]),
    ("trouser",     ["trouser", "track pant", "jogger", "chino", " pant ", "palazzo"]),
    ("jean",        ["jeans", " jean "]),
    ("short",       ["shorts", " short "]),
    ("skirt",       ["skirt"]),
    ("dress",       ["dress", "gown", "maxi"]),
    ("legging",     ["legging", "tights"]),
    ("kurti",       ["kurti", "kurtis"]),
    ("kurta",       ["kurta set", " kurta"]),
    ("saree",       ["saree", " sari"]),
    ("suit",        ["bandhgala suit", "blazer suit", " suit"]),
    ("jacket",      ["jacket", "blazer", "coat "]),
    ("top",         [" top ", "crop top", "tank top", "camisole"]),
    ("shoe",        ["sneaker", "loafer", "boot ", "oxford", "moccasin",
                     "casual shoes", "sports shoes", "running shoes", "walking shoes"]),
    ("sandal",      ["sandal", "slipper", "flip flop", "chappal"]),
    ("heel",        ["heels", "stiletto", "wedge", "platform heel"]),
    ("flat",        ["ballet flat", "flats"]),
    ("backpack",    ["backpack", "rucksack"]),
    ("handbag",     ["handbag", "tote bag", "clutch", "satchel", "shoulder bag"]),
    ("bag",         ["trolley bag", "duffel", "gym bag", "travel bag", " bag"]),
    ("wallet",      ["wallet", "card holder", "card case"]),
    ("watch",       ["watch", "smartwatch"]),
    ("ring",        [" ring", "finger ring"]),
    ("earring",     ["earring", " earrings", " studs", " hoops", "ear ring"]),
    ("necklace",    ["necklace", "chain "]),
    ("bracelet",    ["bracelet", " bangle"]),
    ("bangle",      ["bangle"]),
    ("pendant",     ["pendant"]),
    ("perfume",     ["perfume", "deodorant", "body mist", "fragrance", "deo "]),
    ("sunglasses",  ["sunglass", "sunglasses", "eyewear"]),
    ("cap",         [" cap", " hat ", "beanie"]),
    ("belt",        [" belt"]),
    ("sock",        [" sock", " socks"]),
]


def _classify_type_from_name(name: str) -> str:
    lower = name.lower()
    for canonical, keywords in _NAME_KEYWORD_MAP:
        for kw in keywords:
            if kw in lower:
                return normalize_product_type(canonical)
    return "Unknown"


def derive_catalog_category(product: dict) -> str:
    for field in ("master_category", "category", "sub_category"):
        value = normalize_catalog_category(product.get(field))
        if value != "Unknown":
            return value

    product_type = derive_product_type(product)
    footwear_types = {"Shoe", "Sandal", "Flat", "Heel"}
    accessory_types = {"Backpack", "Handbag", "Bag", "Wallet", "Ring", "Earring",
                       "Bangle", "Bracelet", "Necklace", "Pendant", "Watch",
                       "Sunglasses", "Cap", "Belt", "Sock"}
    personal_care_types = {"Perfume"}

    if product_type in footwear_types:
        return "Shoes & Footwear"
    if product_type in accessory_types:
        return "Accessories"
    if product_type in personal_care_types:
        return "Personal Care"
    if product_type != "Unknown":
        return "Apparel"
    return "Unknown"


def enrich_catalog_products(products: list[dict]) -> list[dict]:
    enriched_products = []
    for product in products:
        enriched = dict(product)
        enriched["image_url"] = normalize_image_url(enriched.get("image_url"), enriched.get("id"))
        enriched["product_type"] = derive_product_type(enriched)
        enriched["category"] = derive_catalog_category(enriched)
        enriched_products.append(enriched)
    return enriched_products


def build_s3_url(key: str) -> str:
    cleaned_key = str(key).replace("\\", "/").lstrip("/")
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{cleaned_key}"


def normalize_image_url(raw_url, product_id=None) -> str:
    candidate = "" if raw_url is None else str(raw_url).strip()
    if candidate.lower() in _INVALID_IMAGE_VALUES:
        candidate = ""

    if candidate.startswith(("http://", "https://")) and S3_BUCKET in candidate:
        # Rewrite old S3 URLs that were saved with the wrong region/endpoint.
        match = re.search(rf"{re.escape(S3_BUCKET)}(?:\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com|\.s3\.amazonaws\.com)/(.*)$", candidate)
        if match:
            return build_s3_url(match.group(1))

    if candidate.startswith(("http://", "https://", "data:image/")):
        return candidate
    if candidate.startswith("//"):
        return f"https:{candidate}"

    if candidate and candidate.isdigit():
        return build_s3_url(f"{candidate}.jpg")

    if candidate:
        cleaned = candidate.replace("\\", "/").strip("/")
        file_name = cleaned.split("/")[-1]
        if "." in file_name:
            return build_s3_url(file_name)

    if product_id is not None:
        return build_s3_url(f"{product_id}.jpg")
    return ""


def cache_busted_image_url(image_url: str) -> str:
    if not image_url:
        return ""
    sep = "&" if "?" in image_url else "?"
    return f"{image_url}{sep}v={CACHE_BUSTER}"


@st.cache_data(ttl=300, show_spinner=False)
def get_s3_products():
    import boto3

    s3 = boto3.client("s3", region_name=S3_REGION)
    bucket_name = S3_BUCKET

    response = s3.list_objects_v2(Bucket=bucket_name)

    # Build a map of product_id -> product metadata from PostgreSQL.
    id_to_product = {}
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                product_id_column = get_products_id_column(cur)
                cur.execute(
                    f"""
                    SELECT
                        {product_id_column},
                        name,
                        price,
                        master_category,
                        sub_category,
                        article_type,
                        gender
                    FROM products
                    """
                )
                id_to_product = {
                    int(row[0]): {
                        "name": str(row[1]) if row[1] is not None else "",
                        "price": row[2],
                        "master_category": row[3],
                        "sub_category": row[4],
                        "article_type": row[5],
                        "gender": row[6],
                    }
                    for row in cur.fetchall()
                }
        finally:
            conn.close()
    except Exception:
        id_to_product = {}

    products = []

    for i, obj in enumerate(response.get("Contents", []), start=1):
        key = obj["Key"]

        if key.endswith((".jpg", ".png", ".jpeg")):
            url = build_s3_url(key)
            base_name = os.path.splitext(os.path.basename(key))[0]

            try:
                lookup_id = int(base_name)
                db_product = id_to_product.get(lookup_id, {})
                display_name = db_product.get("name") or base_name
                product_id = lookup_id
            except ValueError:
                db_product = {}
                display_name = base_name
                product_id = i

            products.append({
                "id": product_id,
                "name": display_name,
                "image_url": url,
                "price": db_product.get("price"),
                "master_category": db_product.get("master_category"),
                "sub_category": db_product.get("sub_category"),
                "article_type": db_product.get("article_type"),
                "gender": db_product.get("gender"),
            })

    return enrich_catalog_products(products)


def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(
            db_url,
            connect_timeout=5,
            options="-c statement_timeout=10000 -c lock_timeout=5000",
        )

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "myntra_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        connect_timeout=5,
        options="-c statement_timeout=10000 -c lock_timeout=5000",
    )


def get_products_id_column(cur) -> str:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'products'
        ORDER BY ordinal_position
        """
    )
    columns = [row[0] for row in cur.fetchall()]
    if "id" in columns:
        return "id"
    if "item_id" in columns:
        return "item_id"
    return "id"


def get_products_columns(cur) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'products'
        """
    )
    return {row[0] for row in cur.fetchall()}


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def init_auth_db() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_login_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_user_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ended_at TIMESTAMPTZ,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_app_user_sessions_user_started
                ON app_user_sessions (user_id, started_at DESC);
                """
            )
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_login_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cart (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id INTEGER,
                    item_id INTEGER NOT NULL,
                    quantity INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute("ALTER TABLE cart ADD COLUMN IF NOT EXISTS session_id INTEGER")
            cur.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.table_constraints
                        WHERE table_schema = 'public'
                          AND table_name = 'cart'
                          AND constraint_name = 'cart_user_id_item_id_key'
                    ) THEN
                        ALTER TABLE cart DROP CONSTRAINT cart_user_id_item_id_key;
                    END IF;
                END $$;
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_cart_user_session_item_unique
                ON cart (user_id, COALESCE(session_id, -1), item_id);
                """
            )
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_recommendations (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    user_email TEXT,
                    session_id INTEGER,
                    item_id INTEGER NOT NULL,
                    source TEXT NOT NULL DEFAULT 'recommendation',
                    shown_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute("ALTER TABLE session_recommendations ALTER COLUMN source SET DEFAULT 'recommendation'")
            cur.execute("ALTER TABLE session_recommendations ADD COLUMN IF NOT EXISTS user_email TEXT")
            cur.execute(
                """
                UPDATE session_recommendations
                SET user_email = user_id
                WHERE user_email IS NULL OR user_email = ''
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_session_recs_unique
                ON session_recommendations (user_id, COALESCE(session_id, -1), item_id, source);
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_session_recs_email_unique
                ON session_recommendations (user_email, COALESCE(session_id, -1), item_id, source);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_recs_user_session
                ON session_recommendations (user_id, session_id, shown_at DESC);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_recs_email_session
                ON session_recommendations (user_email, session_id, shown_at DESC);
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

            cur.execute("SELECT COUNT(*) FROM products")
            should_seed_sample_products = cur.fetchone()[0] == 0

            if not should_seed_sample_products:
                conn.commit()
                return

            product_id_column = get_products_id_column(cur)
            with open("data/products.json", "r", encoding="utf-8") as f:
                seed_products = json.load(f)

            for product in seed_products:
                cur.execute(
                    f"""
                    INSERT INTO products (
                        {product_id_column},
                        name,
                        category,
                        price,
                        image_url,
                        gender,
                        master_category,
                        sub_category,
                        article_type,
                        base_colour,
                        season,
                        product_year
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT ({product_id_column})
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        category = EXCLUDED.category,
                        price = EXCLUDED.price,
                        image_url = EXCLUDED.image_url,
                        gender = EXCLUDED.gender,
                        master_category = EXCLUDED.master_category,
                        sub_category = EXCLUDED.sub_category,
                        article_type = EXCLUDED.article_type,
                        base_colour = EXCLUDED.base_colour,
                        season = EXCLUDED.season,
                        product_year = EXCLUDED.product_year
                    """,
                    (
                        product["id"],
                        product["name"],
                        product["category"],
                        product["price"],
                        product["image_url"],
                        product.get("gender"),
                        product.get("masterCategory") or product.get("master_category"),
                        product.get("subCategory") or product.get("sub_category"),
                        product.get("articleType") or product.get("article_type"),
                        product.get("baseColour") or product.get("base_colour"),
                        product.get("season"),
                        product.get("year") or product.get("product_year"),
                    ),
                )

        conn.commit()
    finally:
        conn.close()


def upsert_user_profile(name: str, email: str) -> None:
    clean_name = name.strip()
    clean_email = email.strip().lower()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '2s'")
            cur.execute("SET LOCAL statement_timeout = '5s'")
            cur.execute(
                """
                INSERT INTO app_users (name, email, last_login_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (email)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    last_login_at = NOW()
                """,
                (clean_name, clean_email),
            )
        conn.commit()
    finally:
        conn.close()


def start_user_session(user_id: str) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE app_user_sessions
                SET is_active = FALSE, ended_at = NOW()
                WHERE user_id = %s AND is_active = TRUE
                """,
                (user_id,),
            )
            cur.execute(
                """
                INSERT INTO app_user_sessions (user_id, started_at, is_active)
                VALUES (%s, NOW(), TRUE)
                RETURNING id
                """,
                (user_id,),
            )
            session_id = int(cur.fetchone()[0])
        conn.commit()
        return session_id
    finally:
        conn.close()


def end_user_session(session_id: int | None) -> None:
    if session_id is None:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE app_user_sessions
                SET is_active = FALSE, ended_at = NOW()
                WHERE id = %s
                """,
                (int(session_id),),
            )
        conn.commit()
    finally:
        conn.close()


def log_recommended_items(
    user_email: str,
    session_id: int | None,
    item_ids: list[int],
    source: str = "recommendation",
) -> None:
    clean_ids = []
    for raw in item_ids:
        try:
            clean_ids.append(int(raw))
        except Exception:
            continue
    if not clean_ids:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO session_recommendations (user_id, user_email, session_id, item_id, source, shown_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING
                """,
                [(user_email, user_email, session_id, item_id, source) for item_id in clean_ids],
            )
        conn.commit()
    finally:
        conn.close()


def fetch_user_sessions(user_id: str, limit: int = 10) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, started_at, ended_at, is_active
                FROM app_user_sessions
                WHERE user_id = %s
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (user_id, int(limit)),
            )
            return list(cur.fetchall())
    finally:
        conn.close()


@st.cache_data(ttl=180, show_spinner=False)
def fetch_products_from_db() -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            product_columns = get_products_columns(cur)
            product_id_column = get_products_id_column(cur)

            id_expr = quote_identifier(product_id_column)

            name_candidates = ["name", "product_name", "title"]
            category_candidates = ["category", "master_category", "sub_category", "article_type"]
            price_candidates = ["price", "discounted_price", "mrp", "base_price"]
            image_candidates = ["image_url", "image", "img_url", "image_path"]

            def text_expr(candidates: list[str], alias: str, default_value: str, *, ignore_unknown: bool = False) -> str:
                available = [quote_identifier(col) for col in candidates if col in product_columns]
                if available:
                    if ignore_unknown:
                        sanitized = [
                            f"NULLIF(NULLIF(TRIM({col}), ''), 'Unknown')" for col in available
                        ]
                        return f"COALESCE({', '.join(sanitized)}, '{default_value}') AS {alias}"
                    return f"COALESCE({', '.join(available)}, '{default_value}') AS {alias}"
                return f"'{default_value}' AS {alias}"

            def first_expr(candidates: list[str], alias: str, default_value: int) -> str:
                for col in candidates:
                    if col in product_columns:
                        return f"{quote_identifier(col)} AS {alias}"
                return f"{default_value} AS {alias}"

            cur.execute(
                "SELECT "
                f"{id_expr} AS id, "
                f"{text_expr(name_candidates, 'name', 'Unknown Product')}, "
                f"{text_expr(category_candidates, 'category', 'Unknown', ignore_unknown=True)}, "
                f"{text_expr(['master_category'], 'master_category', '', ignore_unknown=True)}, "
                f"{text_expr(['sub_category'], 'sub_category', '', ignore_unknown=True)}, "
                f"{text_expr(['article_type'], 'article_type', '', ignore_unknown=True)}, "
                f"{first_expr(price_candidates, 'price', 0)}, "
                f"{text_expr(image_candidates, 'image_url', '')} "
                f"FROM products ORDER BY {id_expr}"
            )
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        products = [dict(zip(columns, row)) for row in rows]
        return enrich_catalog_products(products)
    finally:
        conn.close()


def fetch_events_dataframe() -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM event_logs ORDER BY id DESC"
    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def save_events_to_csv(df: pd.DataFrame, output_path: str = "data/event_logs.csv") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _safe_export_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_", "@", ".", "+") else "_" for ch in str(value))
    return token.strip("_") or "unknown_user"


def _checkout_tracking_csv_path(user_email: str, session_id: int | None = None) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "ranking_pipeline", "recommendation_exports")
    os.makedirs(base_dir, exist_ok=True)
    user_token = _safe_export_token(user_email)
    session_token = f"session_{int(session_id)}" if session_id is not None else "session_unknown"
    return os.path.join(base_dir, f"recommendation_for_{user_token}_{session_token}.csv")


def _write_checkout_csv_with_fallback(df: pd.DataFrame, target_path: str) -> str:
    """
    Write CSV and gracefully handle Windows file locks (e.g. file open in Excel).
    Returns the actual path written.
    """
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        base, ext = os.path.splitext(target_path)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        fallback_path = f"{base}_inuse_{ts}{ext}"
        df.to_csv(fallback_path, index=False)
        return fallback_path


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _compose_tracking_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["user_email"].astype(str)
        + "|"
        + df["session_id"].astype(str)
        + "|"
        + df["source"].astype(str)
        + "|"
        + df["item_id"].astype(str)
    )


def _coerce_optional_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return None


def _build_cart_recommendation_score_lookup(
    cart_rows: list[dict] | None,
) -> dict[int, dict[str, float | None]]:
    if not cart_rows:
        return {}

    cart_qty_by_id: dict[int, int] = {}
    for row in cart_rows:
        try:
            item_id = int(row.get("item_id"))
        except Exception:
            continue

        try:
            qty = int(row.get("quantity", 1) or 1)
        except Exception:
            qty = 1

        cart_qty_by_id[item_id] = cart_qty_by_id.get(item_id, 0) + max(1, qty)

    if not cart_qty_by_id:
        return {}

    sim_df = load_similarity_lookup()
    if sim_df.empty:
        return {}

    scoped = sim_df[sim_df["query_id"].isin(cart_qty_by_id)].copy()
    if scoped.empty:
        return {}

    scoped["similar_id"] = pd.to_numeric(scoped["similar_id"], errors="coerce")
    scoped["similarity_score"] = pd.to_numeric(scoped["similarity_score"], errors="coerce")
    scoped = scoped.dropna(subset=["similar_id", "similarity_score"])
    if scoped.empty:
        return {}

    scoped["similar_id"] = scoped["similar_id"].astype(int)
    scoped = scoped[~scoped["similar_id"].isin(set(cart_qty_by_id))]
    if scoped.empty:
        return {}

    scoped["weight"] = scoped["query_id"].map(lambda q: float(cart_qty_by_id.get(int(q), 1)))
    scoped["weighted_score"] = scoped["similarity_score"] * scoped["weight"]

    agg = scoped.groupby("similar_id", as_index=False).agg(
        weighted_score=("weighted_score", "sum"),
        similarity_score=("similarity_score", "max"),
        total_weight=("weight", "sum"),
    )

    lookup: dict[int, dict[str, float | None]] = {}
    for _, row in agg.iterrows():
        similarity_score = _coerce_optional_float(row.get("similarity_score"))
        weighted_score = _coerce_optional_float(row.get("weighted_score"))
        total_weight = _coerce_optional_float(row.get("total_weight"))
        ranking_score = None
        if weighted_score is not None and total_weight not in (None, 0.0):
            ranking_score = weighted_score / total_weight
        elif similarity_score is not None:
            ranking_score = similarity_score

        try:
            similar_id = int(row["similar_id"])
        except Exception:
            continue

        lookup[similar_id] = {
            "similarity_score": similarity_score,
            "ranking_score": ranking_score,
            "lgbm_ranking_score": None,
            "xgboost_ranking_score": None,
        }

    return lookup


def export_checkout_recommendations_to_csv(
    user_email: str,
    session_id: int | None,
    recommendations: list[dict],
    cart_rows: list[dict] | None = None,
    source: str = "recommendation",
) -> str:
    """Upsert checkout recommendations shown to a per-user CSV tracker."""
    file_path = _checkout_tracking_csv_path(user_email, session_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    score_lookup = _build_cart_recommendation_score_lookup(cart_rows)

    rows = []
    # Add existing cart items first so the CSV captures context before recommendations.
    if cart_rows:
        for cart in cart_rows:
            raw_id = cart.get("item_id")
            try:
                item_id = int(raw_id)
            except Exception:
                continue

            qty = cart.get("quantity")
            try:
                qty_val = int(qty)
            except Exception:
                qty_val = 1

            rows.append(
                {
                    "row_type": "cart_item",
                    "user_email": user_email,
                    "session_id": int(session_id) if session_id is not None else None,
                    "source": "checkout_cart_item",
                    "item_id": item_id,
                    "product_name": str(cart.get("name") or f"item_{item_id}"),
                    "price": cart.get("price"),
                    "quantity": qty_val,
                    "shown_rank": 0,
                    "similarity_score": 0.0,
                    "ranking_score": 0.0,
                    "lgbm_ranking_score": 0.0,
                    "xgboost_ranking_score": 0.0,
                    "added_to_cart": True,
                    "added_to_cart_at": now_iso,
                }
            )

    for idx, rec in enumerate(recommendations, start=1):
        raw_id = rec.get("id")
        try:
            item_id = int(raw_id)
        except Exception:
            continue

        similarity_score = _coerce_optional_float(rec.get("similarity_score"))
        ranking_score = _coerce_optional_float(rec.get("ranking_score"))
        lgbm_ranking_score = _coerce_optional_float(rec.get("lgbm_ranking_score"))
        xgboost_ranking_score = _coerce_optional_float(rec.get("xgboost_ranking_score"))

        if item_id in score_lookup:
            if similarity_score is None:
                similarity_score = score_lookup[item_id].get("similarity_score")
            if ranking_score is None:
                ranking_score = score_lookup[item_id].get("ranking_score")
            if lgbm_ranking_score is None:
                lgbm_ranking_score = score_lookup[item_id].get("lgbm_ranking_score")
            if xgboost_ranking_score is None:
                xgboost_ranking_score = score_lookup[item_id].get("xgboost_ranking_score")

        # Keep CSV schema stable with numeric scores even when a fallback path is used.
        if similarity_score is None:
            similarity_score = 0.0
        if ranking_score is None:
            ranking_score = similarity_score
        if lgbm_ranking_score is None:
            lgbm_ranking_score = ranking_score
        if xgboost_ranking_score is None:
            xgboost_ranking_score = ranking_score

        rows.append(
            {
                "row_type": "recommendation",
                "user_email": user_email,
                "session_id": int(session_id) if session_id is not None else None,
                "source": source,
                "item_id": item_id,
                "product_name": str(rec.get("name") or "Unknown Product"),
                "price": rec.get("price"),
                "quantity": None,
                "shown_rank": idx,
                "similarity_score": similarity_score,
                "ranking_score": ranking_score,
                "lgbm_ranking_score": lgbm_ranking_score,
                "xgboost_ranking_score": xgboost_ranking_score,
                "added_to_cart": False,
                "added_to_cart_at": None,
            }
        )

    if not rows:
        return file_path

    new_df = pd.DataFrame(rows)

    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        existing = _dedupe_columns(existing)
    else:
        existing = pd.DataFrame(columns=new_df.columns)

    # Remove legacy columns so old schema does not leak into fresh exports.
    legacy_cols = [c for c in existing.columns if str(c).strip().lower() in {"last_updated_at", "shown_at"}]
    if legacy_cols:
        existing = existing.drop(columns=legacy_cols)

    for col in new_df.columns:
        if col not in existing.columns:
            existing[col] = None

    # Ensure consistent column set/order for historical files.
    preferred_cols = [
        "row_type",
        "user_email",
        "session_id",
        "source",
        "item_id",
        "product_name",
        "price",
        "quantity",
        "shown_rank",
        "similarity_score",
        "ranking_score",
        "lgbm_ranking_score",
        "xgboost_ranking_score",
        "added_to_cart",
        "added_to_cart_at",
    ]
    for col in preferred_cols:
        if col not in existing.columns:
            existing[col] = None
        if col not in new_df.columns:
            new_df[col] = None
    existing = existing.reindex(columns=[*preferred_cols, *[c for c in existing.columns if c not in preferred_cols]])
    new_df = new_df.reindex(columns=[*preferred_cols, *[c for c in new_df.columns if c not in preferred_cols]])

    replacement_keys = new_df[["user_email", "session_id", "source", "row_type"]].drop_duplicates()
    if not replacement_keys.empty:
        keep_mask = pd.Series(True, index=existing.index)
        for _, repl in replacement_keys.iterrows():
            keep_mask &= ~(
                (existing["user_email"].astype(str) == str(repl["user_email"]))
                & (existing["session_id"].astype(str) == str(repl["session_id"]))
                & (existing["source"].astype(str) == str(repl["source"]))
                & (existing["row_type"].astype(str) == str(repl["row_type"]))
            )
        existing = existing.loc[keep_mask].copy()

    existing["_key"] = _compose_tracking_key(existing)
    new_df["_key"] = _compose_tracking_key(new_df)

    existing = existing.drop_duplicates(subset=["_key"], keep="last").set_index("_key", drop=False)
    new_df = new_df.drop_duplicates(subset=["_key"], keep="last").set_index("_key", drop=False)

    existing.update(new_df)
    appended = new_df.loc[~new_df.index.isin(existing.index)]
    out = pd.concat([existing, appended], ignore_index=True)
    if "_key" in out.columns:
        out = out.drop(columns=["_key"])

    out = out[[c for c in out.columns if str(c).strip().lower() not in {"last_updated_at", "shown_at"}]]

    # Cart rows are context rows and should carry zero scores by requirement.
    if "row_type" in out.columns:
        cart_mask = out["row_type"].astype(str) == "cart_item"
        if "similarity_score" in out.columns:
            out.loc[cart_mask, "similarity_score"] = 0.0
        if "ranking_score" in out.columns:
            out.loc[cart_mask, "ranking_score"] = 0.0
        if "lgbm_ranking_score" in out.columns:
            out.loc[cart_mask, "lgbm_ranking_score"] = 0.0
        if "xgboost_ranking_score" in out.columns:
            out.loc[cart_mask, "xgboost_ranking_score"] = 0.0

    written_path = _write_checkout_csv_with_fallback(out, file_path)
    return written_path


def mark_checkout_recommendation_added_to_cart(
    user_email: str,
    session_id: int | None,
    item_id: int,
    source: str = "recommendation",
) -> str:
    """Mark a shown recommendation as added_to_cart in the per-user CSV tracker."""
    file_path = _checkout_tracking_csv_path(user_email, session_id)
    now_iso = datetime.now(timezone.utc).isoformat()

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = _dedupe_columns(df)
    else:
        df = pd.DataFrame(
            columns=[
                "user_email",
                "session_id",
                "source",
                "item_id",
                "product_name",
                "price",
                "shown_rank",
                "similarity_score",
                "ranking_score",
                "lgbm_ranking_score",
                "xgboost_ranking_score",
                "added_to_cart",
                "added_to_cart_at",
            ]
        )

    for col in ["user_email", "source", "item_id", "session_id", "added_to_cart", "added_to_cart_at", "similarity_score", "ranking_score", "lgbm_ranking_score", "xgboost_ranking_score"]:
        if col not in df.columns:
            df[col] = None

    for legacy in ["last_updated_at", "shown_at"]:
        if legacy in df.columns:
            df = df.drop(columns=[legacy])

    match = (
        (df["user_email"].astype(str) == str(user_email))
        & (df["source"].astype(str) == str(source))
        & (pd.to_numeric(df["item_id"], errors="coerce") == int(item_id))
    )

    if session_id is None:
        match = match & (df["session_id"].isna())
    else:
        match = match & (pd.to_numeric(df["session_id"], errors="coerce") == int(session_id))

    if match.any():
        df.loc[match, "added_to_cart"] = True
        df.loc[match, "added_to_cart_at"] = now_iso
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "user_email": user_email,
                            "session_id": int(session_id) if session_id is not None else None,
                            "source": source,
                            "item_id": int(item_id),
                            "product_name": None,
                            "price": None,
                            "shown_rank": None,
                            "similarity_score": None,
                            "ranking_score": None,
                            "added_to_cart": True,
                            "added_to_cart_at": now_iso,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    written_path = _write_checkout_csv_with_fallback(df, file_path)
    return written_path


def log_event(user_id: str, item_id: int, event_type: str, time_spent_seconds=None):

    event = {
        "user_id": user_id,
        "item_id": item_id,
        "event_type": event_type,
        "time_spent_seconds": time_spent_seconds,
        "timestamp": str(datetime.now(timezone.utc))
    }

    # 1. Send to Kafka (real-time)
    log_event_to_kafka(event)

    # 2. Store in PostgreSQL (permanent)
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


def add_to_cart_db(user_id: str, item_id: int, session_id: int | None = None) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE cart
                SET quantity = quantity + 1
                WHERE user_id = %s
                  AND item_id = %s
                  AND (
                    (session_id IS NULL AND %s IS NULL)
                    OR session_id = %s
                  )
                """,
                (user_id, item_id, session_id, session_id),
            )
            if cur.rowcount == 0:
                cur.execute(
                    """
                    INSERT INTO cart (user_id, session_id, item_id, quantity)
                    VALUES (%s, %s, %s, 1)
                    """,
                    (user_id, session_id, item_id),
                )
        conn.commit()
    finally:
        conn.close()


def fetch_cart(user_id: str, session_id: int | None = None) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if session_id is None:
                cur.execute(
                    """
                    SELECT item_id, quantity
                    FROM cart
                    WHERE user_id = %s AND session_id IS NULL
                    ORDER BY item_id
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT item_id, quantity
                    FROM cart
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY item_id
                    """,
                    (user_id, int(session_id)),
                )
            return list(cur.fetchall())
    finally:
        conn.close()


def fetch_recommendations(user_id: str, products: list[dict]) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT item_id, COUNT(*) AS score
                FROM event_logs
                WHERE user_id = %s AND event_type IN ('click', 'view')
                GROUP BY item_id
                ORDER BY score DESC
                LIMIT 2
                """,
                (user_id,),
            )
            personal = [row["item_id"] for row in cur.fetchall()]

            cur.execute(
                """
                SELECT item_id, COUNT(*) AS score
                FROM event_logs
                WHERE event_type IN ('click', 'view')
                GROUP BY item_id
                ORDER BY score DESC
                LIMIT 4
                """
            )
            global_top = [row["item_id"] for row in cur.fetchall()]
    finally:
        conn.close()

    ranking = personal + [x for x in global_top if x not in personal]
    product_map = {p["id"]: p for p in products}
    return [product_map[item_id] for item_id in ranking if item_id in product_map][:4]


@st.cache_data(ttl=600, show_spinner=False)
def load_similarity_lookup() -> pd.DataFrame:
    root_dir = os.path.dirname(__file__)
    search_dirs = [
        os.path.join(root_dir, "ranking_pipeline"),
        os.path.join(root_dir, "ranking_pipeline", "recommendation_exports"),
        os.path.join(root_dir, "embeddings"),
    ]

    preferred_files = [
        "top5_similar_products_faiss_filtered_simple.csv",
        "top5_similar_products_faiss_filtered.csv",
        "top5_similar_products.csv",
    ]

    required_cols = {"query_id", "similar_id", "similarity_score"}
    seen_paths = set()

    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue

        dir_candidates = list(preferred_files)
        try:
            for fname in sorted(os.listdir(base_dir)):
                if fname.lower().endswith(".csv") and fname not in dir_candidates:
                    dir_candidates.append(fname)
        except Exception:
            continue

        for fname in dir_candidates:
            csv_path = os.path.join(base_dir, fname)
            if csv_path in seen_paths or not os.path.exists(csv_path):
                continue
            seen_paths.add(csv_path)

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            if df.empty:
                continue
            if required_cols.issubset(set(df.columns)):
                return df

    return pd.DataFrame()


def fetch_checkout_recommendations_from_cart(
    cart_rows: list[dict],
    products: list[dict],
    top_k: int = 6,
    user_id: str | None = None,
    events_df: pd.DataFrame | None = None,
) -> list[dict]:
    if not cart_rows or not products:
        return []

    product_map = {int(p["id"]): p for p in products if p.get("id") is not None}
    cart_qty_by_id: dict[int, int] = {}
    for row in cart_rows:
        try:
            item_id = int(row["item_id"])
            qty = int(row.get("quantity", 1) or 1)
            cart_qty_by_id[item_id] = cart_qty_by_id.get(item_id, 0) + max(1, qty)
        except Exception:
            continue

    if not cart_qty_by_id:
        return []

    cart_item_ids = set(cart_qty_by_id.keys())

    cart_categories = set()
    cart_types = set()
    cart_genders = set()
    cart_brands = set()
    non_cart_prices = []
    for cid in cart_item_ids:
        item = product_map.get(cid)
        if not item:
            continue
        cart_categories.add(str(derive_catalog_category(item) or "Unknown").strip().lower())
        cart_types.add(str(item.get("product_type") or derive_product_type(item)).strip().lower())
        cart_genders.add(str(item.get("gender") or "").strip().lower())
        item_name = str(item.get("name") or item.get("product_name") or "")
        if item_name:
            cart_brands.add(item_name.split()[0].strip("-_/.,").lower())

    for pid, item in product_map.items():
        if pid in cart_item_ids:
            continue
        try:
            non_cart_prices.append(float(item.get("price") or 0))
        except Exception:
            pass

    non_cart_prices = [p for p in non_cart_prices if p > 0]
    known_cart_types = {t for t in cart_types if t != "unknown"}
    min_price = min(non_cart_prices) if non_cart_prices else 0.0
    max_price = max(non_cart_prices) if non_cart_prices else 0.0

    def candidate_similarity_score(candidate: dict) -> float:
        score = 0.0
        c_category = str(derive_catalog_category(candidate) or "Unknown").strip().lower()
        c_type = str(candidate.get("product_type") or derive_product_type(candidate)).strip().lower()
        c_gender = str(candidate.get("gender") or "").strip().lower()
        c_name = str(candidate.get("name") or candidate.get("product_name") or "")
        c_brand = c_name.split()[0].strip("-_/.,").lower() if c_name else ""

        # Strongly prioritize same item type over other heuristics.
        if known_cart_types:
            if c_type in known_cart_types:
                score += 8.0
            elif c_type == "unknown":
                score -= 3.0
            else:
                score -= 2.0
        elif c_type in cart_types and c_type != "unknown":
            score += 4.0
        if c_category in cart_categories and c_category != "unknown":
            score += 2.0
        if c_gender and c_gender in cart_genders:
            score += 0.8
        if c_brand and c_brand in cart_brands:
            score += 0.6

        try:
            c_price = float(candidate.get("price") or 0)
            if c_price > 0 and max_price > min_price:
                cheapness = (max_price - c_price) / (max_price - min_price)
                score += 2.0 * max(0.0, min(1.0, cheapness))
            elif c_price > 0 and max_price > 0:
                score += 1.0
        except Exception:
            pass
        return score

    def rank_and_trim(candidates: list[dict]) -> list[dict]:
        filtered = []
        for item in candidates:
            try:
                iid = int(item.get("id"))
            except Exception:
                continue
            if iid in cart_item_ids:
                continue
            filtered.append(item)

        if not filtered:
            return []

        def _sort_score(x: dict) -> float:
            try:
                if x.get("ranking_score") is not None:
                    return float(x.get("ranking_score"))
            except Exception:
                pass
            # Fallback only for ordering, not for exported similarity/ranking fields.
            return candidate_similarity_score(x)

        filtered.sort(key=_sort_score, reverse=True)
        return filtered[:top_k]

    # Try ranking pipeline checkout recommender first.
    try:
        pipeline_path = os.path.join(
            os.path.dirname(__file__),
            "ranking_pipeline",
            "run_pipeline.py",
        )
        spec = importlib.util.spec_from_file_location("ranking_pipeline_run_pipeline", pipeline_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            helper = getattr(module, "recommend_from_cart_items_with_ranker", None)
            if callable(helper):
                products_df = pd.DataFrame(products).copy()
                if "id" in products_df.columns and "item_id" not in products_df.columns:
                    products_df["item_id"] = products_df["id"]

                rank_df = helper(
                    products_df=products_df,
                    cart_item_ids=list(cart_item_ids),
                    user_id=str(user_id or "checkout_user"),
                    events_df=events_df,
                    top_k=top_k,
                )

                if rank_df is None or rank_df.empty:
                    retrieval_helper = getattr(module, "recommend_from_cart_items", None)
                    if callable(retrieval_helper):
                        rank_df = retrieval_helper(
                            products_df=products_df,
                            cart_item_ids=list(cart_item_ids),
                            top_k=top_k,
                        )

                if rank_df is not None and not rank_df.empty:
                    rank_df = rank_df.copy()
                    rank_df["item_id"] = rank_df["item_id"].astype(str)
                    rank_df["similarity_score"] = pd.to_numeric(rank_df.get("similarity_score"), errors="coerce")
                    if "ranking_score" in rank_df.columns:
                        rank_df["ranking_score"] = pd.to_numeric(rank_df.get("ranking_score"), errors="coerce")
                    else:
                        rank_df["ranking_score"] = rank_df["similarity_score"]
                    if "lgbm_ranking_score" in rank_df.columns:
                        rank_df["lgbm_ranking_score"] = pd.to_numeric(rank_df.get("lgbm_ranking_score"), errors="coerce")
                    else:
                        rank_df["lgbm_ranking_score"] = rank_df["ranking_score"]
                    if "xgboost_ranking_score" in rank_df.columns:
                        rank_df["xgboost_ranking_score"] = pd.to_numeric(rank_df.get("xgboost_ranking_score"), errors="coerce")
                    else:
                        rank_df["xgboost_ranking_score"] = rank_df["ranking_score"]

                    score_lookup = {}
                    for _, r in rank_df.iterrows():
                        score_lookup[str(r.get("item_id"))] = {
                            "similarity_score": None if pd.isna(r.get("similarity_score")) else float(r.get("similarity_score")),
                            "ranking_score": None if pd.isna(r.get("ranking_score")) else float(r.get("ranking_score")),
                            "lgbm_ranking_score": None if pd.isna(r.get("lgbm_ranking_score")) else float(r.get("lgbm_ranking_score")),
                            "xgboost_ranking_score": None if pd.isna(r.get("xgboost_ranking_score")) else float(r.get("xgboost_ranking_score")),
                        }

                    rec_ids = []
                    for _, row in rank_df.iterrows():
                        raw_item_id = row.get("item_id")
                        try:
                            rec_ids.append(int(raw_item_id))
                        except Exception:
                            continue

                    recommended = []
                    for rid in rec_ids:
                        item = product_map.get(rid)
                        if item is not None and rid not in cart_item_ids:
                            row_item = dict(item)
                            scores = score_lookup.get(str(rid), {})
                            row_item["similarity_score"] = scores.get("similarity_score")
                            row_item["ranking_score"] = scores.get("ranking_score")
                            row_item["lgbm_ranking_score"] = scores.get("lgbm_ranking_score")
                            row_item["xgboost_ranking_score"] = scores.get("xgboost_ranking_score")
                            recommended.append(row_item)
                        if len(recommended) >= top_k:
                            break

                    if recommended:
                        reranked = rank_and_trim(recommended)
                        if reranked:
                            return reranked
                        return recommended[:top_k]
    except Exception:
        # Fall through to existing similarity logic.
        pass

    sim_df = load_similarity_lookup()

    # Primary strategy: pre-computed embedding similarity from cart items.
    if not sim_df.empty:
        scoped = sim_df[sim_df["query_id"].isin(cart_item_ids)].copy()
        if not scoped.empty:
            scoped["similar_id"] = pd.to_numeric(scoped["similar_id"], errors="coerce")
            scoped["similarity_score"] = pd.to_numeric(scoped["similarity_score"], errors="coerce")
            scoped = scoped.dropna(subset=["similar_id", "similarity_score"])
            scoped["similar_id"] = scoped["similar_id"].astype(int)
            scoped = scoped[~scoped["similar_id"].isin(cart_item_ids)]

            if not scoped.empty:
                scoped["weight"] = scoped["query_id"].map(lambda q: float(cart_qty_by_id.get(int(q), 1)))
                scoped["weighted_score"] = scoped["similarity_score"] * scoped["weight"]

                agg = (
                    scoped.groupby("similar_id", as_index=False).agg(
                        weighted_score=("weighted_score", "sum"),
                        similarity_score=("similarity_score", "max"),
                        total_weight=("weight", "sum"),
                    )
                    .sort_values("weighted_score", ascending=False)
                )

                recommended = []
                for _, row in agg.iterrows():
                    sid = int(row["similar_id"])
                    item = product_map.get(int(sid))
                    if item is not None:
                        row_item = dict(item)
                        row_item["similarity_score"] = float(row["similarity_score"]) if not pd.isna(row["similarity_score"]) else None
                        if not pd.isna(row.get("total_weight")) and float(row.get("total_weight") or 0) > 0:
                            row_item["ranking_score"] = float(row["weighted_score"]) / float(row["total_weight"])
                        else:
                            row_item["ranking_score"] = row_item["similarity_score"]
                        row_item["lgbm_ranking_score"] = row_item["ranking_score"]
                        row_item["xgboost_ranking_score"] = row_item["ranking_score"]
                        recommended.append(row_item)
                    if len(recommended) >= top_k:
                        break
                if recommended:
                    reranked = rank_and_trim(recommended)
                    if reranked:
                        return reranked
                    return recommended[:top_k]

    # Fallback strategy: metadata similarity if pre-computed embedding table misses items.
    scored = []
    for pid, product in product_map.items():
        if pid in cart_item_ids:
            continue
        score = candidate_similarity_score(product)
        row_item = dict(product)
        # Metadata fallback: expose the heuristic score so exports are never blank.
        row_item["similarity_score"] = float(score)
        row_item["ranking_score"] = float(score)
        row_item["lgbm_ranking_score"] = float(score)
        row_item["xgboost_ranking_score"] = float(score)
        scored.append((score, row_item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def flush_active_time(user_id: str) -> None:
    active = st.session_state.get("active_item")
    started_at = st.session_state.get("active_started_at")
    if active is None or started_at is None:
        return

    elapsed = max(0.0, time.time() - started_at)
    log_event(user_id, active, "time_spent", round(elapsed, 2))
    st.session_state["active_item"] = None
    st.session_state["active_started_at"] = None


def set_active_item(user_id: str, item_id: int) -> None:
    current = st.session_state.get("active_item")
    if current is not None:          # flush previous *or* same item before restarting timer
        flush_active_time(user_id)

    st.session_state["active_item"] = item_id
    st.session_state["active_started_at"] = time.time()


def render_product_card(product: dict, user_id: str | None = None, key_prefix: str = "catalog") -> None:
    if user_id is None:
        user_id = st.session_state.get("user_id", "user_1")
    current_session_id = st.session_state.get("current_session_id")

    with st.container(border=True):
        if key_prefix == "catalog":
            st.markdown(f'<div id="catalog-product-{product["id"]}"></div>', unsafe_allow_html=True)
        image_url = normalize_image_url(product.get("image_url"), product.get("id"))
        if image_url:
            st.image(cache_busted_image_url(image_url), width=220)
        else:
            st.caption("Image unavailable")
        product_name = str(product.get("name") or "Unknown Product")
        category = str(product.get("category") or "Unknown")
        product_type = str(product.get("product_type") or derive_product_type(product))
        price = product.get("price")
        try:
            price_text = f"INR {int(float(price))}" if price not in (None, "", 0) else "N/A"
        except (TypeError, ValueError):
            price_text = f"INR {price}" if price not in (None, "") else "N/A"

        st.subheader(product_name)
        st.write(f"Category: {category}")
        if product_type != "Unknown":
            st.write(f"Type: {product_type}")
        st.write(f"Price: {price_text}")

        c1, c2 = st.columns(2)
        if c1.button("View", key=f"{key_prefix}-view-{product['id']}"):
            try:
                item_id = int(product["id"])
                set_active_item(user_id, item_id)
                log_event(user_id, item_id, "view")
                log_event(user_id, item_id, "click")
                st.session_state["last_action_msg"] = f"Viewing {product_name}"
                st.rerun()
            except Exception as exc:
                st.error(f"View failed for {product.get('name', 'item')}: {exc}")

        if c2.button("Add to Cart", key=f"{key_prefix}-cart-{product['id']}"):
            try:
                item_id = int(product["id"])
                flush_active_time(user_id)   # always flush whatever item was being timed
                add_to_cart_db(user_id, item_id, current_session_id)
                log_event(user_id, item_id, "add_to_cart")
                st.session_state["last_action_msg"] = f"Added {product_name} to cart"
                st.session_state["post_add_cart_prompt"] = {
                    "item_id": item_id,
                    "product_name": product_name,
                }
                st.rerun()
            except Exception as exc:
                st.error(f"Add to Cart failed for {product.get('name', 'item')}: {exc}")


def scroll_to_catalog_card(product_id: int) -> None:
    target_id = f"catalog-product-{product_id}"
    components.html(
        f"""
        <script>
        const targetId = "{target_id}";
        const tryScroll = () => {{
            const rootDoc = window.parent && window.parent.document ? window.parent.document : document;
            const el = rootDoc.getElementById(targetId) || document.getElementById(targetId);
            if (el) {{
                el.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
        }};
        setTimeout(tryScroll, 100);
        setTimeout(tryScroll, 400);
        </script>
        """,
        height=0,
    )


@st.cache_resource(show_spinner="Loading product assistant...")
def load_chatbot_agent():
    try:
        from chatbot.agent import build_chatbot_agent
        return build_chatbot_agent()
    except Exception as exc:
        return exc


def _parse_assistant_product_line(line: str) -> tuple[str, int, str] | None:
    match = re.match(r"^\s*[-*]?\s*\[([^\]]+)\]\(\?product_id=(\d+)\)\s*(.*)$", line)
    if not match:
        return None

    label = match.group(1).strip()
    product_id = int(match.group(2))
    details = match.group(3).strip()
    if details.startswith("|"):
        details = details[1:].strip()
    return label, product_id, details


def _activate_assistant_product(product_id: int) -> None:
    st.session_state["highlighted_product_id"] = product_id
    st.session_state["scroll_to_catalog_id"] = product_id
    st.rerun()


def render_chatbot_sidebar() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            min-width: 430px;
            max-width: 430px;
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            border-right: 1px solid #cbd5e1;
        }
        section[data-testid="stSidebar"] .stButton>button {
            border-radius: 12px;
            border: 1px solid #1e293b;
            font-weight: 600;
            background: #ffffff;
        }
        section[data-testid="stSidebar"] .stTextInput input {
            border-radius: 10px;
        }
        .assistant-card {
            border: 1px solid #dbe4ff;
            border-radius: 10px;
            padding: 10px 12px;
            margin: 6px 0;
            background: #ffffff;
        }
        .assistant-role {
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Product Assistant")
    st.sidebar.caption("Ask about products, prices, categories, or similar item recommendations.")

    agent = load_chatbot_agent()

    if isinstance(agent, Exception):
        st.sidebar.error(f"Chatbot unavailable: {agent}")
        return

    if "chat_thread_id" not in st.session_state:
        st.session_state["chat_thread_id"] = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if st.sidebar.button("New Conversation", key="chatbot_reset"):
        st.session_state["chat_messages"] = []
        st.session_state["chat_thread_id"] = str(uuid.uuid4())
        st.rerun()

    chat_history = st.sidebar.container()
    with chat_history:
        for msg_index, msg in enumerate(st.session_state["chat_messages"]):
            role = "You" if msg["role"] == "user" else "Assistant"
            raw = str(msg["content"])
            with st.container(border=True):
                st.markdown(f"<div class=\"assistant-role\">{role}</div>", unsafe_allow_html=True)
                if msg["role"] != "assistant":
                    st.write(raw)
                    continue

                for line_index, line in enumerate(raw.splitlines()):
                    parsed = _parse_assistant_product_line(line)
                    if parsed is None:
                        if line.strip():
                            st.write(line)
                        else:
                            st.markdown("&nbsp;", unsafe_allow_html=True)
                        continue

                    label, product_id, details = parsed
                    if st.button(label, key=f"assistant-product-{msg_index}-{line_index}"):
                        _activate_assistant_product(product_id)
                    if details:
                        st.caption(details)

    with st.sidebar.form("chatbot_form", clear_on_submit=True):
        prompt = st.text_input(
            "Ask about products",
            placeholder="show me shoes, similar to Running Shoes, trending products",
        )
        submitted = st.form_submit_button("Send")

    if submitted and prompt.strip():
        chat_input = prompt.strip()
        st.session_state["chat_messages"].append({"role": "user", "content": chat_input})

        with st.sidebar:
            with st.spinner("Thinking..."):
                try:
                    response = agent.invoke(
                        {"messages": st.session_state["chat_messages"]},
                        config={"configurable": {"thread_id": st.session_state["chat_thread_id"]}},
                    )
                    reply = response["messages"][-1]["content"] if response.get("messages") else "Sorry, I couldn't process that."
                except Exception as exc:
                    reply = f"Error: {exc}"

        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Ecommerce Recommender", layout="wide")
    st.title("Ecommerce Recommender App")
    st.caption("Track views, clicks, and time spent per item in PostgreSQL")

    try:
        init_auth_db()
    except Exception as exc:
        st.error(
            "Unable to connect to PostgreSQL. Set DATABASE_URL or POSTGRES_HOST/PORT/DB/USER/PASSWORD."
        )
        st.exception(exc)
        return

    if "is_logged_in" not in st.session_state:
        st.session_state["is_logged_in"] = False
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = ""
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = ""
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = None

    if not st.session_state["is_logged_in"]:
        st.markdown("### Login")
        st.info("Please log in with your name and email to access the product assistant and catalog.")

        with st.form("login_form"):
            name_input = st.text_input("Name")
            email_input = st.text_input("Email")
            submitted = st.form_submit_button("Continue")

        if submitted:
            name_clean = name_input.strip()
            email_clean = email_input.strip().lower()
            email_ok = re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email_clean)

            if not name_clean:
                st.error("Name is required.")
            elif not email_ok:
                st.error("Enter a valid email address.")
            else:
                try:
                    upsert_user_profile(name_clean, email_clean)
                    session_id = start_user_session(email_clean)
                    st.session_state["is_logged_in"] = True
                    st.session_state["user_name"] = name_clean
                    st.session_state["user_email"] = email_clean
                    st.session_state["user_id"] = email_clean
                    st.session_state["current_session_id"] = session_id
                    st.rerun()
                except Exception as exc:
                    st.warning(
                        "Login succeeded, but we could not persist your profile right now. "
                        "You can continue and retry later."
                    )
                    st.caption(f"Profile DB error: {exc}")
                    st.session_state["is_logged_in"] = True
                    st.session_state["user_name"] = name_clean
                    st.session_state["user_email"] = email_clean
                    st.session_state["user_id"] = email_clean
                    try:
                        st.session_state["current_session_id"] = start_user_session(email_clean)
                    except Exception:
                        st.session_state["current_session_id"] = None
                    st.rerun()
        return

    user_id = st.session_state.get("user_email", "user_1")
    st.session_state["user_id"] = user_id
    if st.session_state.get("current_session_id") is None:
        try:
            st.session_state["current_session_id"] = start_user_session(user_id)
        except Exception:
            st.session_state["current_session_id"] = None

    current_session_id = st.session_state.get("current_session_id")

    hdr_col1, hdr_col2 = st.columns([5, 1])
    with hdr_col1:
        st.caption(f"Logged in as {st.session_state.get('user_name', '')} ({st.session_state.get('user_email', '')})")
        sessions = []
        try:
            sessions = fetch_user_sessions(user_id, limit=6)
        except Exception:
            sessions = []
        if sessions:
            with st.expander("Session History"):
                for s in sessions:
                    sid = s.get("id")
                    started = s.get("started_at")
                    ended = s.get("ended_at")
                    status = "active" if s.get("is_active") else "closed"
                    if ended:
                        st.caption(f"Session {sid}: {started} -> {ended} ({status})")
                    else:
                        st.caption(f"Session {sid}: started {started} ({status})")
    with hdr_col2:
        if st.button("Logout", key="logout_button"):
            flush_active_time(user_id)
            end_user_session(st.session_state.get("current_session_id"))
            st.session_state["is_logged_in"] = False
            st.session_state["user_name"] = ""
            st.session_state["user_email"] = ""
            st.session_state["user_id"] = ""
            st.session_state["current_session_id"] = None
            st.session_state["post_add_cart_prompt"] = None
            st.session_state["checkout_requested"] = False
            st.rerun()

    render_chatbot_sidebar()

    if st.session_state.get("last_action_msg"):
        st.toast(st.session_state["last_action_msg"])
        st.session_state["last_action_msg"] = None

    try:
        init_db()
    except Exception as exc:
        st.warning(f"DB schema sync skipped due to lock/timeout: {exc}")

    if "active_item" not in st.session_state:
        st.session_state["active_item"] = None
    if "active_started_at" not in st.session_state:
        st.session_state["active_started_at"] = None
    if "highlighted_product_id" not in st.session_state:
        st.session_state["highlighted_product_id"] = None
    if "scroll_to_catalog_id" not in st.session_state:
        st.session_state["scroll_to_catalog_id"] = None
    if "post_add_cart_prompt" not in st.session_state:
        st.session_state["post_add_cart_prompt"] = None
    if "checkout_requested" not in st.session_state:
        st.session_state["checkout_requested"] = False

    prompt_payload = st.session_state.get("post_add_cart_prompt")
    if prompt_payload:
        product_name = str(prompt_payload.get("product_name") or "this item")
        st.info(f"{product_name} was added to your cart. What would you like to do next?")
        next_col1, next_col2 = st.columns(2)
        if next_col1.button("Proceed to Checkout", key="post-add-proceed"):
            st.session_state["checkout_requested"] = True
            st.session_state["post_add_cart_prompt"] = None
            st.rerun()
        if next_col2.button("Continue Shopping", key="post-add-continue"):
            st.session_state["checkout_requested"] = False
            st.session_state["post_add_cart_prompt"] = None
            st.rerun()

    try:
        products = fetch_products_from_db()
    except Exception as exc:
        st.warning(f"Could not load products from DB ({exc}). Showing local fallback catalog.")
        products = load_products()

    if not products:
        st.warning("No products found in DB. Showing local fallback catalog.")
        products = load_products()

    cart_rows = fetch_cart(user_id, current_session_id)

    if st.session_state.get("checkout_requested"):
        st.markdown("### View Cart")
        if not cart_rows:
            st.write("Cart is empty.")
        else:
            pmap_checkout = {p["id"]: p for p in products}
            checkout_total = 0
            for row in cart_rows:
                item = pmap_checkout.get(row["item_id"])
                if not item:
                    continue
                subtotal = int(item["price"]) * int(row["quantity"])
                checkout_total += subtotal
                st.write(f"{item['name']} x {row['quantity']} = INR {subtotal}")
            st.success(f"Total: INR {checkout_total}")

        st.markdown("### You might also like")
        checkout_events_df = None
        try:
            checkout_events_df = fetch_events_dataframe()
        except Exception:
            checkout_events_df = None

        checkout_recs = fetch_checkout_recommendations_from_cart(
            cart_rows,
            products,
            user_id=user_id,
            events_df=checkout_events_df,
        )
        checkout_source = "recommendation"
        if checkout_recs:
            log_recommended_items(
                user_email=user_id,
                session_id=current_session_id,
                item_ids=[int(rec.get("id")) for rec in checkout_recs if rec.get("id") is not None],
                source=checkout_source,
            )
            export_checkout_recommendations_to_csv(
                user_email=user_id,
                session_id=current_session_id,
                recommendations=checkout_recs,
                cart_rows=cart_rows,
                source=checkout_source,
            )
            st.caption("Recommended items based on products in your cart")
            rec_cols = st.columns(min(3, len(checkout_recs)))
            for i, rec in enumerate(checkout_recs):
                with rec_cols[i % len(rec_cols)]:
                    image_url = normalize_image_url(rec.get("image_url"), rec.get("id"))
                    if image_url:
                        st.image(cache_busted_image_url(image_url), width=220)
                    else:
                        st.caption("Image unavailable")
                    rec_name = str(rec.get("name") or "Unknown Product")
                    rec_price = rec.get("price")
                    try:
                        rec_price_text = f"INR {int(float(rec_price))}" if rec_price not in (None, "", 0) else "N/A"
                    except (TypeError, ValueError):
                        rec_price_text = f"INR {rec_price}" if rec_price not in (None, "") else "N/A"
                    st.write(rec_name)
                    st.caption(f"Price: {rec_price_text}")
                    rec_item_id = rec.get("id")
                    if rec_item_id is not None:
                        if st.button("Add to Cart", key=f"checkout-rec-cart-{rec_item_id}-{i}"):
                            try:
                                rec_item_id = int(rec_item_id)
                                flush_active_time(user_id)
                                add_to_cart_db(user_id, rec_item_id, current_session_id)
                                log_event(user_id, rec_item_id, "add_to_cart")
                                mark_checkout_recommendation_added_to_cart(
                                    user_email=user_id,
                                    session_id=current_session_id,
                                    item_id=rec_item_id,
                                    source=checkout_source,
                                )
                                st.session_state["last_action_msg"] = f"Added {rec_name} to cart"
                                st.session_state["post_add_cart_prompt"] = {
                                    "item_id": rec_item_id,
                                    "product_name": rec_name,
                                }
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Add to Cart failed for {rec_name}: {exc}")
        else:
            st.info("You might also like recommendations are not available yet. Add a few more items to your cart.")
        st.markdown("---")

    # Highlight a product if navigated from chatbot link
    highlight_id = st.query_params.get("product_id")
    if highlight_id:
        try:
            highlight_id = int(highlight_id)
            st.session_state["highlighted_product_id"] = highlight_id
            st.session_state["scroll_to_catalog_id"] = highlight_id
            # Clear from URL to avoid reloading on every rerun
            st.query_params.clear()
        except (ValueError, TypeError):
            pass
    
    # Display highlighted product from assistant at the top
    if st.session_state["highlighted_product_id"]:
        highlighted_id = st.session_state["highlighted_product_id"]
        highlighted = next((p for p in products if p["id"] == highlighted_id), None)
        if highlighted:
            st.markdown("### 🎯 Selected from Assistant")
            with st.container(border=True):
                col1, col2 = st.columns([2, 3])
                with col1:
                    highlighted_image = normalize_image_url(highlighted.get("image_url"), highlighted.get("id"))
                    if highlighted_image:
                        st.image(cache_busted_image_url(highlighted_image), width=250)
                    else:
                        st.caption("Image unavailable")
                with col2:
                    st.subheader(highlighted["name"])
                    st.write(f"**Category:** {highlighted['category']}")
                    st.write(f"**Price:** INR {highlighted['price']}")
                    c1, c2 = st.columns(2)
                    if c1.button("View", key="highlight-view"):
                        set_active_item(user_id, highlighted["id"])
                        log_event(user_id, highlighted["id"], "view")
                        log_event(user_id, highlighted["id"], "click")
                        st.toast(f"Viewing {highlighted['name']}")
                    if c2.button("Clear", key="highlight-clear"):
                        st.session_state["highlighted_product_id"] = None
                        st.rerun()
            st.markdown("---")

    category_counts = {}
    for p in products:
        cat = normalize_catalog_category(p.get("category"))
        category_counts[cat] = category_counts.get(cat, 0) + 1

    categories = sorted(category_counts.keys())
    st.markdown("### Filter by Category")
    category_filter_keys = []
    for i, category in enumerate(categories):
        key_suffix = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_") or "unknown"
        filter_key = f"catalog_filter_{i}_{key_suffix}"
        category_filter_keys.append((category, filter_key))
        if filter_key not in st.session_state:
            st.session_state[filter_key] = True

    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        if st.button("Select All", key="catalog_select_all"):
            for _, filter_key in category_filter_keys:
                st.session_state[filter_key] = True
            st.rerun()
    with action_col2:
        if st.button("Clear All", key="catalog_clear_all"):
            for _, filter_key in category_filter_keys:
                st.session_state[filter_key] = False
            st.rerun()

    filter_cols = st.columns(min(4, max(1, len(categories))))
    selected_categories = []
    for i, (category, filter_key) in enumerate(category_filter_keys):
        label = f"{category} ({category_counts[category]})"
        with filter_cols[i % len(filter_cols)]:
            checked = st.checkbox(label, key=filter_key)
        if checked:
            selected_categories.append(category)

    if selected_categories:
        st.caption("Active categories: " + ", ".join(selected_categories))

    # Static product-type checkboxes sourced from _TERM_VARIANTS keys
    _ALL_PRODUCT_TYPES = [
        "tshirt", "trouser", "jean", "short", "skirt", "top", "shirt", "jacket",
        "sweatshirt", "backpack", "handbag", "bag", "wallet", "sandal", "heel",
        "flat", "shoe", "earring", "ring", "bangle", "bracelet", "necklace",
        "pendant", "bra", "kurti", "perfume", "watch", "dress", "saree",
        "legging", "suit", "swimwear", "sock", "cap", "belt", "sunglasses",
    ]

    # Count products per canonical type (for display counts next to label)
    product_type_counts = {}
    for product in products:
        pt = str(product.get("product_type") or derive_product_type(product)).lower()
        if pt != "unknown" and pt in _ALL_PRODUCT_TYPES:
            product_type_counts[pt] = product_type_counts.get(pt, 0) + 1

    st.markdown("### Filter by Product Type")
    product_type_filter_keys = []
    for i, pt in enumerate(_ALL_PRODUCT_TYPES):
        filter_key = f"catalog_type_filter_{i}_{pt}"
        product_type_filter_keys.append((pt, filter_key))
        if filter_key not in st.session_state:
            st.session_state[filter_key] = True

    type_action_col1, type_action_col2 = st.columns([1, 1])
    with type_action_col1:
        if st.button("Select All Types", key="catalog_type_select_all"):
            for _, filter_key in product_type_filter_keys:
                st.session_state[filter_key] = True
            st.rerun()
    with type_action_col2:
        if st.button("Clear All Types", key="catalog_type_clear_all"):
            for _, filter_key in product_type_filter_keys:
                st.session_state[filter_key] = False
            st.rerun()

    selected_product_types = []
    type_filter_cols = st.columns(5)
    for i, (pt, filter_key) in enumerate(product_type_filter_keys):
        count = product_type_counts.get(pt, 0)
        label = f"{pt.capitalize()} ({count})"
        with type_filter_cols[i % 5]:
            checked = st.checkbox(label, key=filter_key)
        if checked:
            selected_product_types.append(pt)

    if selected_product_types:
        st.caption("Active types: " + ", ".join(t.capitalize() for t in selected_product_types))

    def matches_active_filters(product: dict) -> bool:
        product_category = normalize_catalog_category(product.get("category"))
        product_type = str(product.get("product_type") or derive_product_type(product)).lower()
        return (product_category in selected_categories) and (
            not selected_product_types or product_type in selected_product_types
        )

    filtered_products = [p for p in products if matches_active_filters(p)]

    if not filtered_products:
        st.info("No products match the selected categories. Select at least one category to view items.")

    if "catalog_page_size" not in st.session_state:
        st.session_state["catalog_page_size"] = 20
    if "catalog_page" not in st.session_state:
        st.session_state["catalog_page"] = 1

    controls_left, controls_right = st.columns([1, 1])
    with controls_left:
        st.selectbox("Items per page", [20, 40, 80], key="catalog_page_size")

    page_size = int(st.session_state["catalog_page_size"])
    total_pages = max(1, math.ceil(len(filtered_products) / page_size))

    target_id = st.session_state.get("scroll_to_catalog_id")
    if target_id and filtered_products:
        target_idx = next((i for i, p in enumerate(filtered_products) if p["id"] == int(target_id)), None)
        if target_idx is not None:
            st.session_state["catalog_page"] = (target_idx // page_size) + 1

    if st.session_state["catalog_page"] > total_pages:
        st.session_state["catalog_page"] = 1

    page_options = list(range(1, total_pages + 1))
    current_page = st.session_state.get("catalog_page", 1)
    if current_page not in page_options:
        current_page = 1

    with controls_right:
        st.selectbox("Page", page_options, index=page_options.index(current_page), key="catalog_page")

    current_page = int(st.session_state["catalog_page"])
    start = (current_page - 1) * page_size
    end = start + page_size
    paged_products = filtered_products[start:end]

    st.caption(f"Showing {len(paged_products)} of {len(filtered_products)} items")

    st.markdown("### Product Catalog")
    cols = st.columns(2)
    for idx, product in enumerate(paged_products):
        with cols[idx % 2]:
            render_product_card(product, user_id, key_prefix="catalog")

    if st.session_state.get("scroll_to_catalog_id"):
        scroll_target = int(st.session_state["scroll_to_catalog_id"])
        if any(p["id"] == scroll_target for p in paged_products):
            scroll_to_catalog_card(scroll_target)
        st.session_state["scroll_to_catalog_id"] = None

    st.subheader("All Images from S3")

    s3_products = get_s3_products()
    filtered_s3_products = [p for p in s3_products if matches_active_filters(p)]

    if not filtered_s3_products:
        st.caption("No S3 products match the selected filters.")

    cols = st.columns(2)  # same layout as your current UI

    for i, product in enumerate(filtered_s3_products):
        with cols[i % 2]:
            render_product_card(product, key_prefix=f"s3-{i}")

    st.markdown("### Recommended For You")
    recs = fetch_recommendations(user_id, products)
    if recs:
        rec_cols = st.columns(len(recs))
        for i, rec in enumerate(recs):
            with rec_cols[i]:
                image_url = normalize_image_url(rec.get("image_url"), rec.get("id"))
                if image_url:
                    st.image(cache_busted_image_url(image_url), width=220)
                else:
                    st.caption("Image unavailable")
                st.write(rec["name"])
    else:
        st.info("Interact with products to generate recommendations.")

    st.markdown("### Cart")
    if not cart_rows:
        st.write("Cart is empty.")
        st.session_state["checkout_requested"] = False
    else:
        pmap = {p["id"]: p for p in products}
        total = 0
        for row in cart_rows:
            item = pmap.get(row["item_id"])
            if not item:
                continue
            subtotal = int(item["price"]) * int(row["quantity"])
            total += subtotal
            st.write(f"{item['name']} x {row['quantity']} = INR {subtotal}")
        st.success(f"Total: INR {total}")

    st.markdown("### Event Logs (DataFrame)")
    df = fetch_events_dataframe()
    st.dataframe(df, use_container_width=True)

    if st.button("Save Event Logs to CSV"):
        csv_path = save_events_to_csv(df)
        st.success(f"Event logs saved to {csv_path}")

    if st.button("Stop Tracking Current Item"):
        flush_active_time(user_id)
        st.toast("Time spent event logged")


if __name__ == "__main__":
    main()

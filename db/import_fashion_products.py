import argparse
import os

import psycopg2
from datasets import load_dataset
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


def ensure_products_schema(cur) -> None:
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
        )
        """
    )
    cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS gender TEXT")
    cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS master_category TEXT")
    cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS sub_category TEXT")
    cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS article_type TEXT")
    cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS base_colour TEXT")
    cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS season TEXT")
    cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS product_year INTEGER")


def build_image_url(product_id: int) -> str:
    bucket_name = os.getenv("S3_BUCKET_NAME", "myntra-project-new")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    return f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{product_id}.jpg"


def import_products(limit: int | None = None) -> int:
    split = f"train[:{limit}]" if limit else "train"
    dataset = load_dataset("ashraq/fashion-product-images-small", split=split)

    conn = get_connection()
    inserted = 0
    try:
        with conn.cursor() as cur:
            ensure_products_schema(cur)

            for index, row in enumerate(dataset, start=1):
                product_id = int(row["id"])
                year_value = row.get("year")
                product_year = int(year_value) if year_value is not None else None

                cur.execute(
                    """
                    INSERT INTO products (
                        id,
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
                    ON CONFLICT (id)
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
                        product_id,
                        row.get("productDisplayName") or row.get("articleType") or f"Product {product_id}",
                        row.get("masterCategory") or "fashion",
                        1000,
                        build_image_url(product_id),
                        row.get("gender"),
                        row.get("masterCategory"),
                        row.get("subCategory"),
                        row.get("articleType"),
                        row.get("baseColour"),
                        row.get("season"),
                        product_year,
                    ),
                )
                inserted += 1

                if index % 500 == 0:
                    conn.commit()
                    print(f"Imported {index} products")

        conn.commit()
        return inserted
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import fashion products into PostgreSQL")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Import only the first N dataset rows for a quick test run.",
    )
    args = parser.parse_args()

    inserted = import_products(limit=args.limit)
    print(f"Imported or updated {inserted} products.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Quick-start script to generate embeddings from products_export.csv.

Usage:
    python quick_start_embeddings.py
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure embeddings module is in path
sys.path.insert(0, str(Path(__file__).parent))

from embeddings.embedding_recommender import EmbeddingRecommender


def main():
    parser = argparse.ArgumentParser(description="Generate product embeddings")
    parser.add_argument(
        "--csv",
        default="data/products_export.csv",
        help="Path to products CSV file",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--image-weight",
        type=float,
        default=0.5,
        help="Weight for image embeddings in combined embedding",
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.5,
        help="Weight for text embeddings in combined embedding",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of products to process (for testing)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EMBEDDING GENERATION STARTED")
    print("=" * 60)

    try:
        # Initialize
        print(f"\n[1/5] Initializing recommender (device={args.device})...")
        recommender = EmbeddingRecommender(device=args.device)

        # Load models
        print("[2/5] Loading CLIP and Sentence-BERT models...")
        recommender.load_models()

        # Load products
        print(f"[3/5] Loading products from {args.csv}...")
        if not os.path.exists(args.csv):
            print(f"[ERROR] CSV file not found: {args.csv}")
            sys.exit(1)

        products = recommender.load_products(args.csv)

        if args.limit:
            products = products.head(args.limit)
            recommender.products_df = products
            print(f"     Limited to {args.limit} products for testing")

        # Generate embeddings
        print(f"[4/5] Generating embeddings (batch_size={args.batch_size})...")
        recommender.generate_product_embeddings(
            batch_size=args.batch_size,
            skip_missing_images=True,
        )

        # Normalize and combine
        print("[5/5] Normalizing and combining embeddings...")
        recommender.normalize_embeddings()
        recommender.combine_embeddings(
            image_weight=args.image_weight,
            text_weight=args.text_weight,
        )

        # Save
        print("\n[SAVE] Saving embeddings to disk...")
        recommender.save_embeddings()

        print("\n" + "=" * 60)
        print("✓ EMBEDDING GENERATION COMPLETED")
        print("=" * 60)

        print("\n[INFO] Embedding statistics:")
        if recommender.image_embeddings is not None:
            print(f"  - Image embeddings: {recommender.image_embeddings.shape}")
        if recommender.text_embeddings is not None:
            print(f"  - Text embeddings: {recommender.text_embeddings.shape}")
        if recommender.combined_embeddings is not None:
            print(f"  - Combined embeddings: {recommender.combined_embeddings.shape}")

        print("\n[NEXT] To use recommendations:")
        print("  1. Python script example:")
        print("     from embeddings.embedding_recommender import EmbeddingRecommender")
        print("     recommender = EmbeddingRecommender()")
        print("     recommender.load_embeddings()")
        print("     recs = recommender.recommend_by_text('blue shirt', top_k=5)")
        print("\n  2. Streamlit app:")
        print("     streamlit run embedding_recommender_app.py")

    except KeyboardInterrupt:
        print("\n[CANCELLED] Embedding generation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

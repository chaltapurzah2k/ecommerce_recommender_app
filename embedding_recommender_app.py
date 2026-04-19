"""
Streamlit app for embedding-based product recommendations.
Uses CLIP + Sentence-BERT embeddings for multimodal similarity search.
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path

# Add embeddings module to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from embeddings.embedding_recommender import EmbeddingRecommender


@st.cache_resource
def load_recommender():
    """Load the embedding recommender (cached)."""
    recommender = EmbeddingRecommender(device="cpu")
    embeddings_path = Path("embeddings/models/product_embeddings.pkl")

    if embeddings_path.exists():
        st.info("Loading pre-computed embeddings...")
        recommender.load_embeddings()
        recommender.load_products("data/products_export.csv")
        return recommender
    else:
        st.warning("Embeddings not found. Please run embedding generation first.")
        return None


def generate_embeddings_ui():
    """UI for generating embeddings."""
    st.subheader("Generate Embeddings")

    if st.button("Generate Product Embeddings from CSV", key="gen_embeddings"):
        with st.spinner("Generating embeddings... This may take several minutes."):
            try:
                recommender = EmbeddingRecommender(device="cpu")
                recommender.load_models()
                recommender.load_products("data/products_export.csv")
                recommender.generate_product_embeddings(batch_size=32, skip_missing_images=True)
                recommender.normalize_embeddings()
                recommender.combine_embeddings(image_weight=0.5, text_weight=0.5)
                recommender.save_embeddings()
                st.success("Embeddings generated and saved successfully!")
            except Exception as e:
                st.error(f"Error generating embeddings: {e}")


def text_search_ui(recommender):
    """UI for text-based recommendations."""
    st.subheader("Search by Product Description")

    query_text = st.text_input("Enter product description", placeholder="e.g., blue shirt for men")

    if query_text and st.button("Find Similar Products"):
        with st.spinner("Searching..."):
            recommendations = recommender.recommend_by_text(query_text, top_k=8)

            if recommendations:
                st.success(f"Found {len(recommendations)} similar products:")

                cols = st.columns(2)
                for idx, rec in enumerate(recommendations):
                    with cols[idx % 2]:
                        with st.container(border=True):
                            st.write(f"**Score:** {rec['similarity_score']:.4f}")
                            st.write(f"**ID:** {rec['id']}")
                            st.write(f"**Name:** {rec['name']}")
                            st.write(f"**Category:** {rec.get('category', 'N/A')}")
                            st.write(f"**Gender:** {rec.get('gender', 'N/A')}")
                            st.write(f"**Color:** {rec.get('base_colour', 'N/A')}")

                            if rec.get("image_url"):
                                st.image(rec["image_url"], use_column_width=True)
            else:
                st.warning("No recommendations found.")


def product_search_ui(recommender):
    """UI for finding similar products."""
    st.subheader("Find Similar Products")

    product_id = st.number_input(
        "Enter Product ID",
        min_value=1,
        max_value=int(recommender.products_df["id"].max()),
        value=None,
        placeholder="e.g., 1163",
    )

    if product_id and st.button("Get Recommendations"):
        with st.spinner("Finding similar products..."):
            product = recommender.products_df[recommender.products_df["id"] == product_id]

            if len(product) > 0:
                st.write("### Selected Product")
                p = product.iloc[0]
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Name:** {p['name']}")
                        st.write(f"**Category:** {p.get('category', 'N/A')}")
                        st.write(f"**Gender:** {p.get('gender', 'N/A')}")
                    with col2:
                        if p.get("image_url"):
                            st.image(p["image_url"], use_column_width=True)

                recommendations = recommender.get_product_recommendations_by_id(product_id, top_k=8)

                if recommendations:
                    st.write("### Similar Products")

                    cols = st.columns(2)
                    for idx, rec in enumerate(recommendations):
                        with cols[idx % 2]:
                            with st.container(border=True):
                                st.write(f"**Score:** {rec['similarity_score']:.4f}")
                                st.write(f"**ID:** {rec['id']}")
                                st.write(f"**Name:** {rec['name']}")
                                st.write(f"**Category:** {rec.get('category', 'N/A')}")
                                st.write(f"**Price:** INR {rec.get('price', 'N/A')}")

                                if rec.get("image_url"):
                                    st.image(rec["image_url"], use_column_width=True)
            else:
                st.error(f"Product ID {product_id} not found.")


def stats_ui(recommender):
    """UI for embeddings statistics."""
    st.subheader("Embedding Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Products", len(recommender.products_df))

    with col2:
        img_count = recommender.image_embeddings.shape[0] if recommender.image_embeddings is not None else 0
        st.metric("Products with Image Embeddings", img_count)

    with col3:
        text_count = recommender.text_embeddings.shape[0] if recommender.text_embeddings is not None else 0
        st.metric("Products with Text Embeddings", text_count)

    with col4:
        if recommender.combined_embeddings is not None:
            st.metric("Combined Embeddings", recommender.combined_embeddings.shape[0])
        else:
            st.metric("Combined Embeddings", 0)

    if recommender.image_embeddings is not None:
        st.write(f"Image Embedding Dimension: {recommender.image_embeddings.shape[1]}")

    if recommender.text_embeddings is not None:
        st.write(f"Text Embedding Dimension: {recommender.text_embeddings.shape[1]}")


def main():
    st.set_page_config(
        page_title="Embedding-based Recommendations",
        layout="wide",
    )

    st.title("🔍 Embedding-based Product Recommendations")
    st.write("Using CLIP (image) + Sentence-BERT (text) embeddings for multimodal similarity search")

    # Initialize recommender
    recommender = load_recommender()

    if recommender is None:
        st.warning("Embeddings not generated yet. Please generate them first.")
        generate_embeddings_ui()
    else:
        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Generate Embeddings", "Search by Text", "Find Similar Products", "Statistics"]
        )

        with tab1:
            generate_embeddings_ui()

        with tab2:
            text_search_ui(recommender)

        with tab3:
            product_search_ui(recommender)

        with tab4:
            stats_ui(recommender)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Test script for hybrid retrieval system (BM25 + semantic + cross-encoder re-ranking)
"""

import os
import time
import sys

# Set up path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.hybrid_retrieval import load_hybrid_retriever
from sentence_transformers import SentenceTransformer
import numpy as np


def test_hybrid_retrieval():
    """Test hybrid retrieval with sample queries."""
    
    print("\n" + "="*60)
    print("HYBRID RETRIEVAL TEST")
    print("="*60)
    
    try:
        # Load retriever
        embeddings_path = "embeddings/models/product_embeddings.pkl"
        if not os.path.exists(embeddings_path):
            print(f"❌ Embeddings file not found: {embeddings_path}")
            print("Run: python quick_start_embeddings.py")
            return
        
        print("\n[1] Loading hybrid retriever...")
        retriever = load_hybrid_retriever(embeddings_path, device="cpu")
        text_encoder = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        print("✓ Retriever loaded")
        
        # Test queries
        test_queries = [
            ("running shoes", "Simple product search"),
            ("blue shirt under 2000", "Query with budget filter"),
            ("womens jeans", "Category + gender search"),
            ("red dress online", "Color + product type"),
            ("casual shoes", "Style search"),
        ]
        
        print("\n[2] Testing hybrid search queries...\n")
        
        for query, description in test_queries:
            print(f"Query: '{query}' ({description})")
            print("-" * 50)
            
            # Encode query
            query_embedding = text_encoder.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            # Measure time
            start = time.time()
            
            # Hybrid search
            results = retriever.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=5,
                candidate_k=50,
                bm25_weight=0.4,
                semantic_weight=0.6,
                use_reranker=True,
            )
            
            elapsed = (time.time() - start) * 1000
            
            if results:
                print(f"✓ Found {len(results)} results in {elapsed:.0f}ms\n")
                for r in results[:3]:  # Show top 3
                    print(f"  Rank {r['rank']}: {r['name']}")
                    print(f"    Category: {r['category']} | Price: INR {r['price']}")
                    print(f"    BM25: {r['bm25_score']:.2f} | Semantic: {r['semantic_score']:.2f}")
                    print(f"    Hybrid: {r['hybrid_score']:.2f} | Rerank: {r['rerank_score']:.2f}")
                    print()
            else:
                print(f"✗ No results found\n")
        
        # Compare retrieval methods
        print("\n[3] Comparing retrieval methods...")
        print("-" * 50)
        
        test_query = "red running shoes"
        query_embedding = text_encoder.encode(test_query, convert_to_numpy=True)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # BM25 only
        start = time.time()
        bm25_results = retriever.bm25_search(test_query, k=5)
        bm25_time = (time.time() - start) * 1000
        print(f"\nBM25 only: {len(bm25_results)} results in {bm25_time:.1f}ms")
        
        # Semantic only
        start = time.time()
        semantic_results = retriever.semantic_search(query_embedding, k=5)
        semantic_time = (time.time() - start) * 1000
        print(f"Semantic only: {len(semantic_results)} results in {semantic_time:.1f}ms")
        
        # Hybrid
        start = time.time()
        hybrid_results = retriever.hybrid_search(
            query=test_query,
            query_embedding=query_embedding,
            top_k=5,
            use_reranker=False,  # Without re-ranking
        )
        hybrid_time = (time.time() - start) * 1000
        print(f"Hybrid (no rerank): {len(hybrid_results)} results in {hybrid_time:.1f}ms")
        
        # Hybrid with re-ranking
        start = time.time()
        hybrid_rerank_results = retriever.hybrid_search(
            query=test_query,
            query_embedding=query_embedding,
            top_k=5,
            use_reranker=True,
        )
        hybrid_rerank_time = (time.time() - start) * 1000
        print(f"Hybrid + rerank: {len(hybrid_rerank_results)} results in {hybrid_rerank_time:.1f}ms")
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hybrid_retrieval()

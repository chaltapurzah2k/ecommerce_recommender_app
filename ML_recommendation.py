import os

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

try:
    import redis
except Exception:
    redis = None

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


@st.cache_data(ttl=10)
def fetch_events() -> pd.DataFrame:
    conn = get_connection()
    try:
        query = """
            SELECT
                id,
                user_id,
                item_id,
                event_type,
                time_spent_seconds,
                event_time
            FROM event_logs
            ORDER BY event_time DESC
        """
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def build_scored_events(df: pd.DataFrame) -> pd.DataFrame:
    event_weight = {
        "view": 1,
        "click": 2,
        "add_to_cart": 3,
        "time_spent": 2,
    }
    scored = df.copy()
    scored["score"] = scored["event_type"].map(event_weight).fillna(0)
    return scored


def build_user_item_matrix(scored_df: pd.DataFrame) -> pd.DataFrame:
    return scored_df.pivot_table(
        index="user_id",
        columns="item_id",
        values="score",
        aggfunc="sum",
        fill_value=0,
    )


def build_similarity_df(user_item: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(user_item)
    return pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)


def recommend(user_id: str, user_item: pd.DataFrame, similarity_df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
    items = user_item.loc[similar_users.index].sum().sort_values(ascending=False)
    seen_items = user_item.loc[user_id]
    recommendations = items[seen_items == 0]
    return recommendations.head(top_n)


def store_recommendations_in_redis(user_id: str, recs: pd.Series) -> tuple[bool, str]:
    if redis is None:
        return False, "redis package is not installed."

    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        for item, score in recs.items():
            r.zadd(f"recs:{user_id}", {str(int(item)): float(score)})
        return True, f"Stored {len(recs)} recommendations in recs:{user_id}"
    except Exception as exc:
        return False, f"Redis write failed: {exc}"


def main() -> None:
    st.set_page_config(page_title="ML Recommendation Pipeline", layout="wide")
    st.title("ML Recommendation Pipeline From PostgreSQL")

    try:
        df = fetch_events()
    except Exception as exc:
        st.error("Failed to fetch event logs from PostgreSQL.")
        st.exception(exc)
        return

    if df.empty:
        st.warning("No rows found in event_logs.")
        return

    scored = build_scored_events(df)
    user_item = build_user_item_matrix(scored)

    if user_item.empty:
        st.info("Not enough data to build user-item matrix.")
        return

    similarity_df = build_similarity_df(user_item)

    st.markdown("### Events with Scores")
    st.dataframe(scored[["user_id", "item_id", "event_type", "score", "event_time"]], use_container_width=True)

    st.markdown("### User-Item Matrix")
    st.dataframe(user_item, use_container_width=True)

    st.markdown("### User Similarity Matrix")
    st.dataframe(similarity_df, use_container_width=True)

    selected_user = st.selectbox("Select User", options=user_item.index.astype(str).tolist())
    top_n = st.slider("Top N Recommendations", min_value=1, max_value=10, value=5)

    recs = recommend(selected_user, user_item, similarity_df, top_n=top_n)
    if recs.empty:
        st.info("No recommendations available for this user.")
    else:
        rec_df = recs.reset_index()
        rec_df.columns = ["item_id", "score"]
        st.markdown("### Recommended Items")
        st.dataframe(rec_df, use_container_width=True)

        if st.button("Store Recommendations in Redis"):
            ok, msg = store_recommendations_in_redis(selected_user, recs)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)


if __name__ == "__main__":
    main()

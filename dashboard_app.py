import os
from datetime import date, timedelta

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
                e.id,
                e.user_id,
                e.item_id,
                COALESCE(p.name, CONCAT('Item ', e.item_id::text)) AS product_name,
                e.event_type,
                e.time_spent_seconds,
                e.event_time
            FROM event_logs e
            LEFT JOIN products p ON p.id = e.item_id
            ORDER BY e.event_time DESC
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
        df = df[df["event_time"].notna()].copy()
        df["user_id"] = df["user_id"].astype(str)
        df["event_date"] = df["event_time"].dt.date

    return df


def apply_filters(df: pd.DataFrame, selected_users: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    filtered = df.copy()
    if selected_users:
        filtered = filtered[filtered["user_id"].isin(selected_users)]

    if not filtered.empty:
        start_ts = pd.Timestamp(start_date).tz_localize("UTC")
        end_ts = pd.Timestamp(end_date + timedelta(days=1)).tz_localize("UTC")
        filtered = filtered[
            (filtered["event_time"] >= start_ts) &
            (filtered["event_time"] < end_ts)
        ]

    return filtered


def clamp_date(value: date, min_value: date, max_value: date) -> date:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def build_kpis(df: pd.DataFrame) -> dict:
    views = int((df["event_type"] == "view").sum()) if not df.empty else 0
    clicks = int((df["event_type"] == "click").sum()) if not df.empty else 0
    add_to_cart = int((df["event_type"] == "add_to_cart").sum()) if not df.empty else 0

    time_df = df[(df["event_type"] == "time_spent") & (df["time_spent_seconds"].notna())] if not df.empty else pd.DataFrame()
    avg_time = float(time_df["time_spent_seconds"].mean()) if not time_df.empty else 0.0

    conversion_rate = (add_to_cart / views * 100.0) if views > 0 else 0.0

    return {
        "views": views,
        "clicks": clicks,
        "add_to_cart": add_to_cart,
        "avg_time": avg_time,
        "conversion_rate": conversion_rate,
    }


def product_event_counts(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["product_name", "count"])

    grouped = (
        df[df["event_type"] == event_type]
        .groupby("product_name", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    return grouped


def time_spent_by_product(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["product_name", "avg_time_spent_seconds"])

    tdf = df[(df["event_type"] == "time_spent") & (df["time_spent_seconds"].notna())]
    if tdf.empty:
        return pd.DataFrame(columns=["product_name", "avg_time_spent_seconds"])

    grouped = (
        tdf.groupby("product_name", as_index=False)["time_spent_seconds"]
        .mean()
        .rename(columns={"time_spent_seconds": "avg_time_spent_seconds"})
        .sort_values("avg_time_spent_seconds", ascending=False)
    )
    return grouped


def conversion_by_product(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["product_name", "views", "add_to_cart", "conversion_rate_pct"])

    views = (
        df[df["event_type"] == "view"]
        .groupby("product_name", as_index=False)
        .size()
        .rename(columns={"size": "views"})
    )
    carts = (
        df[df["event_type"] == "add_to_cart"]
        .groupby("product_name", as_index=False)
        .size()
        .rename(columns={"size": "add_to_cart"})
    )

    merged = pd.merge(views, carts, on="product_name", how="outer").fillna(0)
    merged["views"] = merged["views"].astype(int)
    merged["add_to_cart"] = merged["add_to_cart"].astype(int)
    merged["conversion_rate_pct"] = merged.apply(
        lambda row: (row["add_to_cart"] / row["views"] * 100.0) if row["views"] > 0 else 0.0,
        axis=1,
    )

    return merged.sort_values("conversion_rate_pct", ascending=False)


def build_scored_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

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
    if scored_df.empty:
        return pd.DataFrame()

    return scored_df.pivot_table(
        index="user_id",
        columns="item_id",
        values="score",
        aggfunc="sum",
        fill_value=0,
    )


def build_similarity_df(user_item: pd.DataFrame) -> pd.DataFrame:
    if user_item.empty:
        return pd.DataFrame()

    similarity = cosine_similarity(user_item)
    return pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)


def recommend(user_id: str, user_item: pd.DataFrame, similarity_df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    if user_item.empty or similarity_df.empty or user_id not in similarity_df.columns:
        return pd.Series(dtype="float64")

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
    if similar_users.empty:
        return pd.Series(dtype="float64")

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
        return True, f"Stored {len(recs)} recommendations in Redis key recs:{user_id}"
    except Exception as exc:
        return False, f"Redis write failed: {exc}"


def main() -> None:
    st.set_page_config(page_title="Ecommerce Analytics Dashboard", layout="wide")
    st.title("Ecommerce Analytics Dashboard")
    st.caption("PostgreSQL -> Pandas -> Streamlit")

    try:
        df = fetch_events()
    except Exception as exc:
        st.error("Unable to read event logs from PostgreSQL.")
        st.exception(exc)
        return

    if df.empty:
        st.warning("No event data found in event_logs yet.")
        return

    st.sidebar.header("Filters")
    all_users = sorted(df["user_id"].dropna().astype(str).unique().tolist())
    selected_users = st.sidebar.multiselect("User IDs", options=all_users, default=all_users)

    min_date = df["event_date"].min()
    max_date = df["event_date"].max()
    today = date.today()
    min_selectable = min(min_date, today - timedelta(days=365))
    max_selectable = max(max_date, today)

    if "start_date" not in st.session_state:
        st.session_state["start_date"] = min_date
    else:
        st.session_state["start_date"] = clamp_date(st.session_state["start_date"], min_selectable, max_selectable)

    if "end_date" not in st.session_state:
        st.session_state["end_date"] = max_date
    else:
        st.session_state["end_date"] = clamp_date(st.session_state["end_date"], min_selectable, max_selectable)

    if st.sidebar.button("Reset Date Range", use_container_width=True):
        st.session_state["start_date"] = min_date
        st.session_state["end_date"] = max_date

    date_col1, date_col2 = st.sidebar.columns(2)
    with date_col1:
        start_date = st.date_input(
            "Start Date",
            min_value=min_selectable,
            max_value=max_selectable,
            key="start_date"
        )

    if st.session_state["end_date"] < start_date:
        st.session_state["end_date"] = start_date

    with date_col2:
        end_date = st.date_input(
            "End Date",
            min_value=start_date,
            max_value=max_selectable,
            key="end_date"
        )

    if end_date < start_date:
        st.session_state["end_date"] = start_date
        end_date = start_date

    filtered = apply_filters(df, selected_users, start_date, end_date)

    st.sidebar.caption(f"Data available from {min_date} to {max_date}.")
    st.sidebar.caption(f"Filtered events: {len(filtered)}")

    kpis = build_kpis(filtered)

    # Display metrics in highlighted container
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Views", kpis["views"])
        c2.metric("Total Clicks", kpis["clicks"])
        c3.metric("Total Add-to-Cart", kpis["add_to_cart"])
        c4.metric("Avg Time Spent (s)", f"{kpis['avg_time']:.2f}")
        c5.metric("Conversion Rate", f"{kpis['conversion_rate']:.2f}%")

    st.markdown("### Most Viewed Products")
    viewed = product_event_counts(filtered, "view")
    if viewed.empty:
        st.info("No view events in current filter.")
    else:
        st.bar_chart(viewed.set_index("product_name")["count"])
        st.dataframe(viewed, use_container_width=True)

    st.markdown("### Most Clicked Products")
    clicked = product_event_counts(filtered, "click")
    if clicked.empty:
        st.info("No click events in current filter.")
    else:
        st.bar_chart(clicked.set_index("product_name")["count"])
        st.dataframe(clicked, use_container_width=True)

    st.markdown("### Funnel: View -> Click -> Cart")
    funnel_df = pd.DataFrame(
        {
            "stage": ["view", "click", "add_to_cart"],
            "count": [kpis["views"], kpis["clicks"], kpis["add_to_cart"]],
        }
    )
    st.bar_chart(funnel_df.set_index("stage")["count"])
    st.dataframe(funnel_df, use_container_width=True)

    st.markdown("### Time Spent Per Product")
    tdf = time_spent_by_product(filtered)
    if tdf.empty:
        st.info("No time_spent events in current filter.")
    else:
        st.bar_chart(tdf.set_index("product_name")["avg_time_spent_seconds"])
        st.dataframe(tdf, use_container_width=True)

    st.markdown("### Conversion Rate by Product")
    conv = conversion_by_product(filtered)
    if conv.empty:
        st.info("No product conversion data in current filter.")
    else:
        st.dataframe(conv, use_container_width=True)

    st.markdown("### Raw Event Table")
    st.dataframe(filtered, use_container_width=True)

    st.markdown("### Recommender Pipeline From PostgreSQL")
    scored = build_scored_events(filtered)
    user_item = build_user_item_matrix(scored)
    similarity_df = build_similarity_df(user_item)

    if user_item.empty:
        st.info("Not enough data to build user-item matrix.")
    else:
        st.write("Scored Events (with weights)")
        st.dataframe(scored[["user_id", "item_id", "event_type", "score", "event_time"]], use_container_width=True)

        st.write("User-Item Matrix")
        st.dataframe(user_item, use_container_width=True)

        st.write("User Similarity Matrix")
        st.dataframe(similarity_df, use_container_width=True)

        selected_user = st.selectbox("Select User for Recommendations", options=user_item.index.astype(str).tolist())
        top_n = st.slider("Top N Recommendations", min_value=1, max_value=10, value=5)

        recs = recommend(selected_user, user_item, similarity_df, top_n=top_n)
        if recs.empty:
            st.info("No recommendations available for the selected user.")
        else:
            rec_df = recs.reset_index()
            rec_df.columns = ["item_id", "score"]
            st.dataframe(rec_df, use_container_width=True)

            if st.button("Store Recommendations in Redis"):
                ok, msg = store_recommendations_in_redis(selected_user, recs)
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Events CSV",
        data=csv_data,
        file_name="filtered_event_logs.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

"""
Snowflake Cost & Usage Monitor

A multi-tab dashboard tracking:
- Overall costs and credit consumption
- Compute (warehouse) usage
- AI & serverless services
- Storage costs
- DDL/DML query auditing
"""

from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Snowflake Monitor",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# Connection
# =============================================================================


def _is_running_in_snowflake() -> bool:
    """Detect if app is running inside Snowflake (SiS)."""
    try:
        from snowflake.snowpark.context import get_active_session
        get_active_session()
        return True
    except Exception:
        return False


def get_session():
    """Return a Snowpark session. Works both locally and in Snowflake."""
    if _is_running_in_snowflake():
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    else:
        try:
            return st.connection("snowflake").session()
        except Exception as e:
            st.error(f"Failed to connect to Snowflake: {e}")
            st.info(
                "Configure your connection in `.streamlit/secrets.toml`."
            )
            st.stop()


def run_query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Execute SQL and return a pandas DataFrame."""
    session = get_session()
    if params:
        # Replace positional :1, :2 etc. with actual values for Snowpark
        query = sql
        for i, val in enumerate(params, 1):
            query = query.replace(f":{i}", str(val))
        df = session.sql(query).to_pandas()
    else:
        df = session.sql(sql).to_pandas()
    df.columns = df.columns.str.lower()
    return df


# =============================================================================
# Data Loading
# =============================================================================

LOOKBACK_OPTIONS = {"7 days": 7, "14 days": 14, "30 days": 30, "90 days": 90}


@st.cache_data(ttl=600, show_spinner="Loading overview data...")
def load_daily_credits(days: int) -> pd.DataFrame:
    return run_query(
        """
        SELECT
            SERVICE_TYPE,
            USAGE_DATE,
            CREDITS_USED,
            CREDITS_BILLED
        FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
        WHERE USAGE_DATE >= DATEADD(DAY, -:1, CURRENT_DATE())
        ORDER BY USAGE_DATE
        """,
        params=[days],
    )


@st.cache_data(ttl=600, show_spinner="Loading warehouse data...")
def load_warehouse_credits(days: int) -> pd.DataFrame:
    return run_query(
        """
        SELECT
            WAREHOUSE_NAME,
            DATE_TRUNC('day', START_TIME)::DATE AS usage_date,
            ROUND(SUM(CREDITS_USED), 2) AS credits_used
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
        WHERE START_TIME >= DATEADD(DAY, -:1, CURRENT_DATE())
        GROUP BY WAREHOUSE_NAME, DATE_TRUNC('day', START_TIME)::DATE
        ORDER BY usage_date, credits_used DESC
        """,
        params=[days],
    )


@st.cache_data(ttl=600, show_spinner="Loading warehouse comparison...")
def load_warehouse_comparison() -> pd.DataFrame:
    return run_query(
        """
        WITH recent AS (
            SELECT warehouse_name, SUM(credits_used) AS credits
            FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
            WHERE start_time >= DATEADD(day, -14, CURRENT_DATE())
            GROUP BY warehouse_name
        ),
        prior AS (
            SELECT warehouse_name, SUM(credits_used) AS credits
            FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
            WHERE start_time >= DATEADD(day, -28, CURRENT_DATE())
                AND start_time < DATEADD(day, -14, CURRENT_DATE())
            GROUP BY warehouse_name
        )
        SELECT
            COALESCE(r.warehouse_name, p.warehouse_name) AS warehouse_name,
            ROUND(COALESCE(p.credits, 0), 2) AS prior_14d,
            ROUND(COALESCE(r.credits, 0), 2) AS recent_14d,
            ROUND(COALESCE(r.credits, 0) - COALESCE(p.credits, 0), 2) AS change,
            ROUND(
                ((COALESCE(r.credits, 0) - COALESCE(p.credits, 0))
                 / NULLIF(p.credits, 0)) * 100, 1
            ) AS pct_change
        FROM recent r
        FULL OUTER JOIN prior p ON r.warehouse_name = p.warehouse_name
        ORDER BY ABS(COALESCE(r.credits, 0) - COALESCE(p.credits, 0)) DESC
        LIMIT 15
        """
    )


@st.cache_data(ttl=600, show_spinner="Loading serverless task data...")
def load_serverless_tasks(days: int) -> pd.DataFrame:
    return run_query(
        """
        SELECT
            TASK_NAME,
            DATABASE_NAME,
            SCHEMA_NAME,
            ROUND(SUM(CREDITS_USED), 4) AS total_credits,
            COUNT(*) AS run_count
        FROM SNOWFLAKE.ACCOUNT_USAGE.SERVERLESS_TASK_HISTORY
        WHERE START_TIME >= DATEADD(DAY, -:1, CURRENT_DATE())
        GROUP BY TASK_NAME, DATABASE_NAME, SCHEMA_NAME
        ORDER BY total_credits DESC
        """,
        params=[days],
    )


@st.cache_data(ttl=600, show_spinner="Loading storage data...")
def load_storage() -> pd.DataFrame:
    return run_query(
        """
        SELECT
            DATABASE_NAME,
            USAGE_DATE,
            ROUND(AVERAGE_DATABASE_BYTES / POW(1024, 3), 2) AS database_gb,
            ROUND(AVERAGE_FAILSAFE_BYTES / POW(1024, 3), 2) AS failsafe_gb,
            ROUND((AVERAGE_DATABASE_BYTES + AVERAGE_FAILSAFE_BYTES
                   + COALESCE(AVERAGE_HYBRID_TABLE_STORAGE_BYTES, 0))
                  / POW(1024, 3), 2) AS total_gb
        FROM SNOWFLAKE.ACCOUNT_USAGE.DATABASE_STORAGE_USAGE_HISTORY
        WHERE USAGE_DATE >= DATEADD(DAY, -30, CURRENT_DATE())
        ORDER BY USAGE_DATE DESC
        """
    )


@st.cache_data(ttl=600, show_spinner="Loading query audit data...")
def load_query_audit(days: int) -> pd.DataFrame:
    return run_query(
        """
        SELECT
            QUERY_TYPE,
            USER_NAME,
            DATABASE_NAME,
            SCHEMA_NAME,
            WAREHOUSE_NAME,
            EXECUTION_STATUS,
            START_TIME,
            TOTAL_ELAPSED_TIME,
            ROWS_PRODUCED,
            LEFT(QUERY_TEXT, 200) AS query_preview
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
        WHERE START_TIME >= DATEADD(DAY, -:1, CURRENT_DATE())
            AND QUERY_TYPE IN (
                'CREATE_TABLE', 'CREATE_TABLE_AS_SELECT', 'CREATE_VIEW',
                'ALTER_TABLE', 'DROP_TABLE', 'DROP_VIEW',
                'INSERT', 'UPDATE', 'DELETE', 'MERGE',
                'TRUNCATE_TABLE', 'COPY'
            )
        ORDER BY START_TIME DESC
        LIMIT 1000
        """,
        params=[days],
    )


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.header("⚙️ Filters")
    lookback_label = st.selectbox("Lookback period", list(LOOKBACK_OPTIONS.keys()), index=2)
    lookback_days = LOOKBACK_OPTIONS[lookback_label]
    st.divider()
    st.caption("Data from `SNOWFLAKE.ACCOUNT_USAGE`")
    st.caption("Latency: up to 45 min")


# =============================================================================
# Tabs
# =============================================================================

tab_overview, tab_compute, tab_ai, tab_storage, tab_audit = st.tabs(
    [
        "📈 Overview",
        "⚡ Compute",
        "🤖 AI & Services",
        "💾 Storage",
        "🔍 Query Audit",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: Overview
# -----------------------------------------------------------------------------

with tab_overview:
    daily_df = load_daily_credits(lookback_days)

    if daily_df.empty:
        st.info("No metering data found for the selected period.")
    else:
        total_credits = daily_df["credits_used"].sum()
        total_billed = daily_df["credits_billed"].sum()
        top_service = (
            daily_df.groupby("service_type")["credits_used"]
            .sum()
            .idxmax()
        )

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Credits Used", f"{total_credits:,.2f}")
        kpi2.metric("Total Credits Billed", f"{total_billed:,.2f}")
        kpi3.metric("Top Service", top_service)

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.subheader("Credits by Service Type")
                service_agg = (
                    daily_df.groupby("service_type", as_index=False)["credits_used"]
                    .sum()
                    .sort_values("credits_used", ascending=False)
                )
                chart = (
                    alt.Chart(service_agg)
                    .mark_bar()
                    .encode(
                        x=alt.X("credits_used:Q", title="Credits Used"),
                        y=alt.Y("service_type:N", sort="-x", title="Service Type"),
                        color=alt.Color("service_type:N", legend=None),
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart, use_container_width=True)

        with col2:
            with st.container():
                st.subheader("Daily Credit Trend")
                trend = (
                    daily_df.groupby("usage_date", as_index=False)["credits_used"]
                    .sum()
                )
                chart = (
                    alt.Chart(trend)
                    .mark_area(opacity=0.6)
                    .encode(
                        x=alt.X("usage_date:T", title="Date"),
                        y=alt.Y("credits_used:Q", title="Credits"),
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart, use_container_width=True)

        with st.container():
            st.subheader("Daily Breakdown by Service")
            st.dataframe(
                daily_df.pivot_table(
                    index="usage_date",
                    columns="service_type",
                    values="credits_used",
                    aggfunc="sum",
                )
                .fillna(0)
                .sort_index(ascending=False),
                use_container_width=True,
            )

# -----------------------------------------------------------------------------
# Tab 2: Compute
# -----------------------------------------------------------------------------

with tab_compute:
    wh_df = load_warehouse_credits(lookback_days)

    if wh_df.empty:
        st.info("No warehouse metering data found.")
    else:
        # Top warehouses
        wh_totals = (
            wh_df.groupby("warehouse_name", as_index=False)["credits_used"]
            .sum()
            .sort_values("credits_used", ascending=False)
        )

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(
            "Total Warehouse Credits",
            f"{wh_totals['credits_used'].sum():,.2f}",
        )
        kpi2.metric(
            "Active Warehouses",
            str(wh_totals.shape[0]),
        )
        if not wh_totals.empty:
            kpi3.metric(
                "Top Warehouse",
                wh_totals.iloc[0]["warehouse_name"],
                f"{wh_totals.iloc[0]['credits_used']:,.2f} credits",
            )

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.subheader("Top Warehouses by Credits")
                chart = (
                    alt.Chart(wh_totals.head(10))
                    .mark_bar()
                    .encode(
                        x=alt.X("credits_used:Q", title="Credits"),
                        y=alt.Y("warehouse_name:N", sort="-x", title="Warehouse"),
                        color=alt.Color("warehouse_name:N", legend=None),
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

        with col2:
            with st.container():
                st.subheader("Daily Warehouse Trend")
                daily_wh = (
                    wh_df.groupby("usage_date", as_index=False)["credits_used"].sum()
                )
                chart = (
                    alt.Chart(daily_wh)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("usage_date:T", title="Date"),
                        y=alt.Y("credits_used:Q", title="Credits"),
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

        with st.container():
            st.subheader("Warehouse Comparison: Recent 14d vs Prior 14d")
            comp_df = load_warehouse_comparison()
            if comp_df.empty:
                st.info("Not enough history for comparison.")
            else:
                st.dataframe(
                    comp_df,
                    use_container_width=True,
                    
                )

# -----------------------------------------------------------------------------
# Tab 3: AI & Services
# -----------------------------------------------------------------------------

with tab_ai:
    daily_df_ai = load_daily_credits(lookback_days)

    if daily_df_ai.empty:
        st.info("No service metering data found.")
    else:
        ai_df = daily_df_ai[daily_df_ai["service_type"] == "AI_SERVICES"]
        clustering_df = daily_df_ai[daily_df_ai["service_type"] == "AUTO_CLUSTERING"]

        kpi1, kpi2 = st.columns(2)
        ai_total = ai_df["credits_used"].sum() if not ai_df.empty else 0
        cluster_total = clustering_df["credits_used"].sum() if not clustering_df.empty else 0
        kpi1.metric("AI Services Credits", f"{ai_total:,.4f}")
        kpi2.metric("Auto-Clustering Credits", f"{cluster_total:,.4f}")

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.subheader("AI Services Daily Trend")
                if ai_df.empty:
                    st.info("No AI service usage in this period.")
                else:
                    chart = (
                        alt.Chart(ai_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("usage_date:T", title="Date"),
                            y=alt.Y("credits_used:Q", title="Credits"),
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(chart, use_container_width=True)

        with col2:
            with st.container():
                st.subheader("Auto-Clustering Daily Trend")
                if clustering_df.empty:
                    st.info("No auto-clustering usage in this period.")
                else:
                    chart = (
                        alt.Chart(clustering_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("usage_date:T", title="Date"),
                            y=alt.Y("credits_used:Q", title="Credits"),
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(chart, use_container_width=True)

        with st.container():
            st.subheader("Serverless Tasks")
            tasks_df = load_serverless_tasks(lookback_days)
            if tasks_df.empty:
                st.info("No serverless task usage in this period.")
            else:
                st.dataframe(tasks_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Tab 4: Storage
# -----------------------------------------------------------------------------

with tab_storage:
    storage_df = load_storage()

    if storage_df.empty:
        st.info("No storage data found.")
    else:
        # Latest snapshot per database
        latest_date = storage_df["usage_date"].max()
        latest = storage_df[storage_df["usage_date"] == latest_date].copy()
        total_storage = latest["total_gb"].sum()
        total_failsafe = latest["failsafe_gb"].sum()

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Storage", f"{total_storage:,.2f} GB")
        kpi2.metric("Failsafe Storage", f"{total_failsafe:,.2f} GB")
        kpi3.metric("Databases", str(latest.shape[0]))

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.subheader("Top Databases by Storage")
                top_dbs = latest.sort_values("total_gb", ascending=False).head(10)
                chart = (
                    alt.Chart(top_dbs)
                    .mark_bar()
                    .encode(
                        x=alt.X("total_gb:Q", title="Total GB"),
                        y=alt.Y("database_name:N", sort="-x", title="Database"),
                        color=alt.Color("database_name:N", legend=None),
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

        with col2:
            with st.container():
                st.subheader("Storage Composition (Latest)")
                composition = pd.DataFrame(
                    {
                        "type": ["Active Data", "Failsafe"],
                        "gb": [
                            latest["database_gb"].sum(),
                            latest["failsafe_gb"].sum(),
                        ],
                    }
                )
                chart = (
                    alt.Chart(composition)
                    .mark_arc(innerRadius=50)
                    .encode(
                        theta=alt.Theta("gb:Q"),
                        color=alt.Color("type:N", title="Type"),
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

        with st.container():
            st.subheader("Storage Trend (Top 10 Databases, Last 30 Days)")
            top_db_names = (
                latest.sort_values("total_gb", ascending=False)
                .head(10)["database_name"]
                .tolist()
            )
            trend_df = storage_df[storage_df["database_name"].isin(top_db_names)]
            if not trend_df.empty:
                chart = (
                    alt.Chart(trend_df)
                    .mark_line()
                    .encode(
                        x=alt.X("usage_date:T", title="Date"),
                        y=alt.Y("total_gb:Q", title="Total GB"),
                        color=alt.Color("database_name:N", title="Database"),
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 5: Query Audit
# -----------------------------------------------------------------------------

with tab_audit:
    audit_df = load_query_audit(lookback_days)

    if audit_df.empty:
        st.info("No DDL/DML queries found in this period.")
    else:
        # Summary KPIs
        total_queries = len(audit_df)
        unique_users = audit_df["user_name"].nunique()
        failed = (audit_df["execution_status"] != "SUCCESS").sum()

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("DDL/DML Queries", str(total_queries))
        kpi2.metric("Unique Users", str(unique_users))
        kpi3.metric("Failed Queries", str(failed))

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.subheader("Queries by Type")
                type_counts = (
                    audit_df["query_type"]
                    .value_counts()
                    .reset_index()
                )
                type_counts.columns = ["query_type", "count"]
                chart = (
                    alt.Chart(type_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Count"),
                        y=alt.Y("query_type:N", sort="-x", title="Query Type"),
                        color=alt.Color("query_type:N", legend=None),
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart, use_container_width=True)

        with col2:
            with st.container():
                st.subheader("Top Users (DDL/DML)")
                user_counts = (
                    audit_df["user_name"]
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                user_counts.columns = ["user_name", "count"]
                chart = (
                    alt.Chart(user_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Count"),
                        y=alt.Y("user_name:N", sort="-x", title="User"),
                        color=alt.Color("user_name:N", legend=None),
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart, use_container_width=True)

        with st.container():
            st.subheader("Recent DDL/DML Queries")

            # Filters
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                type_filter = st.multiselect(
                    "Query Type",
                    audit_df["query_type"].unique().tolist(),
                    default=None,
                )
            with filter_col2:
                user_filter = st.multiselect(
                    "User",
                    audit_df["user_name"].unique().tolist(),
                    default=None,
                )
            with filter_col3:
                status_filter = st.multiselect(
                    "Status",
                    audit_df["execution_status"].unique().tolist(),
                    default=None,
                )

            filtered = audit_df.copy()
            if type_filter:
                filtered = filtered[filtered["query_type"].isin(type_filter)]
            if user_filter:
                filtered = filtered[filtered["user_name"].isin(user_filter)]
            if status_filter:
                filtered = filtered[filtered["execution_status"].isin(status_filter)]

            st.dataframe(
                filtered,
                use_container_width=True,
                
            )

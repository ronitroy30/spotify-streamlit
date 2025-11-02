import os
import datetime as dt
import pandas as pd
import streamlit as st
import altair as alt
from dateutil.relativedelta import relativedelta
from athena import read_sql  # uses get_engine() inside athena.py

# ---------- App config ----------
SCHEMA = st.secrets.get("ATHENA_SCHEMA", os.getenv("ATHENA_SCHEMA", "spotify_analytics"))
st.set_page_config(page_title="Spotify Analytics — dbt + Athena", layout="wide")
st.title("🎧 Spotify Analytics — dbt + Athena")

# ---------- Controls ----------
default_end = dt.date.today()
default_start = default_end - relativedelta(months=1)
col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    start = st.date_input("Start date", value=default_start, key="start_date")
with col_b:
    end = st.date_input("End date", value=default_end, key="end_date")
with col_c:
    st.info("Athena credentials & settings come from Streamlit Secrets. Data reads your dbt-built tables.", icon="ℹ️")

dedupe = st.toggle("Dedupe plays (distinct events)", value=True, help="Removes duplicate events before counting")

# ---------- Schema helpers ----------
@st.cache_data(ttl=300)
def get_table_columns(schema: str, table: str):
    q = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{schema}'
      AND table_name   = '{table}'
    ORDER BY ordinal_position
    """
    try:
        df = read_sql(q)
        return set(df["column_name"].str.lower().tolist())
    except Exception:
        return set()

def plays_time_expr(cols: set) -> str:
    # Prefer real event timestamps if present; otherwise fallback to dt
    if "played_at" in cols:
        return "played_at"
    if "played_at_ts" in cols:
        return "played_at_ts"
    return "dt"

@st.cache_data(ttl=300)
def plays_cols() -> set:
    return get_table_columns(SCHEMA, "stg_spotify__plays")

@st.cache_data(ttl=300)
def table_exists(schema: str, table: str) -> bool:
    q = f"""
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = '{schema}'
      AND table_name   = '{table}'
    LIMIT 1
    """
    try:
        df = read_sql(q)
        return not df.empty
    except Exception:
        return False

# ---------- Loaders (schema-aware, dedupe-aware, time-aware) ----------
@st.cache_data(ttl=300)
def load_kpis(start_date, end_date, dedupe_events: bool):
    s, e = str(start_date), str(end_date)
    cols = plays_cols()
    time_col = plays_time_expr(cols)

    if time_col in ("played_at", "played_at_ts"):
        distinct_kw = "DISTINCT " if dedupe_events else ""
        sql = f"""
        WITH dedup AS (
          SELECT {distinct_kw}{time_col}, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE DATE({time_col}) BETWEEN DATE '{s}' AND DATE '{e}'
        ),
        daily AS (
          SELECT CAST(date({time_col}) AS date) AS dt,
                 COUNT(*) AS plays,
                 COUNT(DISTINCT track_id) AS unique_tracks
          FROM dedup GROUP BY 1
        ),
        totals AS (
          SELECT
            SUM(plays) AS total_plays,
            SUM(unique_tracks) AS total_unique_tracks,
            AVG(plays) AS avg_plays_per_day,
            COUNT(*) AS active_days
          FROM daily
        )
        SELECT * FROM totals
        """
    else:
        distinct_kw = "DISTINCT " if dedupe_events else ""
        sql = f"""
        WITH dedup AS (
          SELECT {distinct_kw}dt, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE dt BETWEEN DATE '{s}' AND DATE '{e}'
        ),
        daily AS (
          SELECT CAST(dt AS date) AS dt,
                 COUNT(*) AS plays,
                 COUNT(DISTINCT track_id) AS unique_tracks
          FROM dedup GROUP BY 1
        ),
        totals AS (
          SELECT
            SUM(plays) AS total_plays,
            SUM(unique_tracks) AS total_unique_tracks,
            AVG(plays) AS avg_plays_per_day,
            COUNT(*) AS active_days
          FROM daily
        )
        SELECT * FROM totals
        """
    return read_sql(sql)

@st.cache_data(ttl=300)
def load_daily_series(start_date, end_date, dedupe_events: bool):
    s, e = str(start_date), str(end_date)
    cols = plays_cols()
    time_col = plays_time_expr(cols)

    if time_col in ("played_at", "played_at_ts"):
        distinct_kw = "DISTINCT " if dedupe_events else ""
        sql = f"""
        WITH dedup AS (
          SELECT {distinct_kw}{time_col}, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE DATE({time_col}) BETWEEN DATE '{s}' AND DATE '{e}'
        )
        SELECT CAST(date({time_col}) AS date) AS dt,
               COUNT(*) AS plays,
               COUNT(DISTINCT track_id) AS unique_tracks
        FROM dedup
        GROUP BY 1
        ORDER BY 1
        """
    else:
        distinct_kw = "DISTINCT " if dedupe_events else ""
        sql = f"""
        WITH dedup AS (
          SELECT {distinct_kw}dt, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE dt BETWEEN DATE '{s}' AND DATE '{e}'
        )
        SELECT CAST(dt AS date) AS dt,
               COUNT(*) AS plays,
               COUNT(DISTINCT track_id) AS unique_tracks
        FROM dedup
        GROUP BY 1
        ORDER BY 1
        """
    return read_sql(sql)

@st.cache_data(ttl=300)
def load_top_tracks(start_date, end_date, dedupe_events: bool, limit=15):
    s, e = str(start_date), str(end_date)
    cols = plays_cols()
    time_col = plays_time_expr(cols)

    if time_col in ("played_at", "played_at_ts"):
        distinct_kw = "DISTINCT " if dedupe_events else ""
        dedup_cte = f"""
        WITH dedup AS (
          SELECT {distinct_kw}{time_col}, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE DATE({time_col}) BETWEEN DATE '{s}' AND DATE '{e}'
        )
        """
    else:
        distinct_kw = "DISTINCT " if dedupe_events else ""
        dedup_cte = f"""
        WITH dedup AS (
          SELECT {distinct_kw}dt, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE dt BETWEEN DATE '{s}' AND DATE '{e}'
        )
        """

    # Try nice version with dim_tracks join
    try:
        sql_join = f"""
        {dedup_cte}
        SELECT d.track_id,
               COALESCE(t.track_name, d.track_id) AS track_name,
               COUNT(*) AS play_count
        FROM dedup d
        LEFT JOIN {SCHEMA}.dim_tracks t
          ON d.track_id = t.track_id
        GROUP BY 1,2
        ORDER BY play_count DESC
        LIMIT {int(limit)}
        """
        df = read_sql(sql_join)
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback without dim
    sql_simple = f"""
    {dedup_cte}
    SELECT d.track_id,
           d.track_id AS track_name,
           COUNT(*) AS play_count
    FROM dedup d
    GROUP BY 1,2
    ORDER BY play_count DESC
    LIMIT {int(limit)}
    """
    return read_sql(sql_simple)

@st.cache_data(ttl=300)
def load_sessions(start_date, end_date):
    s, e = str(start_date), str(end_date)
    if not table_exists(SCHEMA, "fct_listening_sessions"):
        return pd.DataFrame(columns=["session_id","session_start","session_end","track_plays","session_duration_seconds"])
    sql = f"""
    SELECT session_id, session_start, session_end, track_plays, session_duration_seconds
    FROM {SCHEMA}.fct_listening_sessions
    WHERE DATE(session_start) BETWEEN DATE '{s}' AND DATE '{e}'
    ORDER BY session_start DESC
    LIMIT 1000
    """
    return read_sql(sql)

# ---------- KPI tiles ----------
try:
    kpis = load_kpis(start, end, dedupe)
    c1, c2, c3, c4 = st.columns(4)
    if not kpis.empty:
        r = kpis.iloc[0].fillna(0)
        c1.metric("Total Plays", int(r.get("total_plays", 0) or 0))
        c2.metric("Unique Tracks", int(r.get("total_unique_tracks", 0) or 0))
        c3.metric("Avg Plays / Day", round(float(r.get("avg_plays_per_day", 0) or 0), 2))
        c4.metric("Active Days", int(r.get("active_days", 0) or 0))

        # Top track caption (safe)
        try:
            tops_for_caption = load_top_tracks(start, end, dedupe, limit=1)
            if not tops_for_caption.empty:
                top = tops_for_caption.iloc[0]
                st.caption(f"Top track: **{top['track_name']}** ({int(top['play_count'])} plays)")
            else:
                st.caption("Top track: — (0 plays)")
        except Exception:
            st.caption("Top track: — (0 plays)")
    else:
        st.warning("No data for the selected range.")
except Exception as e:
    st.error("Failed to load KPIs from Athena. See Diagnostics above.")
    st.exception(e)

st.divider()

# ---------- Trends ----------
try:
    daily = load_daily_series(start, end, dedupe)
    left, right = st.columns([2, 1])
    with left:
        if not daily.empty:
            chart = alt.Chart(daily).mark_line(point=True).encode(
                x=alt.X("dt:T", title="Date"),
                y=alt.Y("plays:Q", title="Plays"),
                tooltip=["dt:T", "plays:Q", "unique_tracks:Q"]
            ).properties(height=280, title="Plays per day")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No daily data yet.")
    with right:
        if not daily.empty:
            st.dataframe(
                daily.rename(columns={"dt": "Date", "plays": "Plays", "unique_tracks": "Unique tracks"}),
                use_container_width=True,
                height=280
            )
except Exception as e:
    st.error("Failed to load daily series.")
    st.exception(e)

st.divider()

# ---------- Top tracks ----------
try:
    tops = load_top_tracks(start, end, dedupe, limit=15)
    if not tops.empty:
        bars = alt.Chart(tops).mark_bar().encode(
            x=alt.X("play_count:Q", title="Plays"),
            y=alt.Y("track_name:N", sort="-x", title="Track"),
            tooltip=["track_name:N", "play_count:Q"]
        ).properties(height=400, title="Top tracks")
        st.altair_chart(bars, use_container_width=True)
        st.dataframe(tops, use_container_width=True, hide_index=True)
    else:
        st.info("No top tracks in this range.")
except Exception as e:
    st.error("Failed to load top tracks.")
    st.exception(e)

st.divider()

# ---------- Sessions ----------
st.subheader("Recent listening sessions (30-min gaps)")
try:
    sessions = load_sessions(start, end)
    if not sessions.empty:
        sessions = sessions.copy()
        sessions["duration_min"] = (sessions["session_duration_seconds"].astype(float) / 60).round(1)
        st.dataframe(
            sessions[["session_id", "session_start", "session_end", "track_plays", "duration_min"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No sessions found in this range.")
except Exception as e:
    st.error("Failed to load sessions.")
    st.exception(e)

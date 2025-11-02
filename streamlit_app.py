import os, datetime as dt
import pandas as pd
import streamlit as st
import altair as alt
from dateutil.relativedelta import relativedelta
from athena import read_sql

SCHEMA = st.secrets.get("ATHENA_SCHEMA", os.getenv("ATHENA_SCHEMA", "spotify_analytics"))

st.set_page_config(page_title="Spotify Analytics — dbt + Athena", layout="wide")
st.title("🎧 Spotify Analytics — dbt + Athena")
import boto3, pandas as pd, streamlit as st
from sqlalchemy.engine import create_engine

with st.expander("🔧 Live diagnostics (temporary)"):
    ok = True
    # 1) STS identity
    try:
        region = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
        sts = boto3.client("sts", region_name=region)
        ident = sts.get_caller_identity()
        st.success(f"STS OK → Account: {ident['Account']}  |  Arn: {ident['Arn']}")
    except Exception as e:
        ok = False; st.error("❌ STS failed (bad/expired keys or wrong region)"); st.exception(e)

    # 2) Athena SELECT current_date (workgroup/output/encryption sanity)
    if ok:
        try:
            workgroup = st.secrets.get("ATHENA_WORKGROUP", "spotify_analytics_wg")
            stage = st.secrets["ATHENA_S3_STAGING"]  # must end with /
            conn = (f"awsathena+rest://@athena.{region}.amazonaws.com/"
                    f"awsdatacatalog?work_group={workgroup}&s3_staging_dir={stage}")
            eng = create_engine(conn)
            with eng.connect() as con:
                df = pd.read_sql("SELECT current_date AS today", con=con)
            st.success(f"Athena SELECT OK → {df.iloc[0]['today']}")
        except Exception as e:
            ok = False; st.error("❌ Athena SELECT failed"); st.exception(e)

    # 3) List tables in your Glue DB/schema
    if ok:
        try:
            schema = st.secrets.get("ATHENA_SCHEMA", "spotify_analytics")
            with eng.connect() as con:
                t = pd.read_sql(f"SHOW TABLES IN {schema}", con=con)
            st.write("Tables in schema:", t)
            if t.empty:
                st.warning("Schema has no tables. You must create/crawl/CTAS/dbt them.")
        except Exception as e:
            ok = False; st.error("❌ SHOW TABLES failed (wrong schema name or Glue perms)"); st.exception(e)

    # 4) Smoke query against a table you use
    if ok:
        try:
            schema = st.secrets.get("ATHENA_SCHEMA", "spotify_analytics")
            q = f"SELECT count(*) AS c FROM {schema}.stg_spotify__plays LIMIT 1"
            with eng.connect() as con:
                c = pd.read_sql(q, con=con)
            st.success(f"stg_spotify__plays exists → count sample: {int(c.iloc[0]['c'])}")
        except Exception as e:
            st.error("❌ Could not read table stg_spotify__plays")
            st.info("Likely causes: table doesn’t exist in this schema, no partitions, or wrong region/workgroup.")
            st.exception(e)


# Date pickers
default_end = dt.date.today()
default_start = default_end - relativedelta(months=1)
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    start = st.date_input("Start date", value=default_start, key="start_date")
with col_b:
    end = st.date_input("End date", value=default_end, key="end_date")
with col_c:
    st.info("Athena credentials & settings come from Streamlit Secrets. Data reads your dbt-built tables.", icon="ℹ️")
    
@st.cache_data(ttl=300)
def table_exists(schema: str, table: str) -> bool:
    from athena import read_sql
    q = f"""
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = '{schema}'
      AND table_name = '{table}'
    LIMIT 1
    """
    try:
        df = read_sql(q)
        return not df.empty
    except Exception:
        return False
# ---- Schema helpers ----
@st.cache_data(ttl=300)
def get_table_columns(schema: str, table: str):
    from athena import read_sql
    q = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{schema}'
      AND table_name = '{table}'
    ORDER BY ordinal_position
    """
    try:
        df = read_sql(q)
        return set(df["column_name"].str.lower().tolist())
    except Exception:
        return set()

def plays_time_expr(cols: set):
    # Decide which timestamp/date column to use for daily grouping
    # and dedupe key.
    if "played_at" in cols:
        return "played_at"
    if "played_at_ts" in cols:
        return "played_at_ts"
    # Fallback: no explicit timestamp, use partition/date column
    return "dt"

@st.cache_data(ttl=300)
def plays_cols():
    return get_table_columns(SCHEMA, "stg_spotify__plays")

@st.cache_data(ttl=300)
def load_kpis(start_date, end_date):
    s, e = str(start_date), str(end_date)
    cols = plays_cols()
    time_col = plays_time_expr(cols)

    if time_col in ("played_at", "played_at_ts"):
        # Use timestamp column for dedupe and daily grouping
        sql = f"""
        WITH dedup AS (
          SELECT DISTINCT {time_col}, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE dt BETWEEN DATE '{s}' AND DATE '{e}'
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
        # No timestamp column → dedupe and group by dt (partition/date)
        sql = f"""
        WITH dedup AS (
          SELECT DISTINCT dt, track_id
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
def load_daily_series(start_date, end_date):
    s, e = str(start_date), str(end_date)
    cols = plays_cols()
    time_col = plays_time_expr(cols)

    if time_col in ("played_at", "played_at_ts"):
        sql = f"""
        WITH dedup AS (
          SELECT DISTINCT {time_col}, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE dt BETWEEN DATE '{s}' AND DATE '{e}'
        )
        SELECT CAST(date({time_col}) AS date) AS dt,
               COUNT(*) AS plays,
               COUNT(DISTINCT track_id) AS unique_tracks
        FROM dedup
        GROUP BY 1
        ORDER BY 1
        """
    else:
        sql = f"""
        WITH dedup AS (
          SELECT DISTINCT dt, track_id
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
def load_top_tracks(start_date, end_date, limit=15):
    s, e = str(start_date), str(end_date)
    cols = plays_cols()
    time_col = plays_time_expr(cols)

    # Build a dedup CTE that matches available columns
    if time_col in ("played_at", "played_at_ts"):
        dedup_cte = f"""
        WITH dedup AS (
          SELECT DISTINCT {time_col}, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE dt BETWEEN DATE '{s}' AND DATE '{e}'
        )
        """
        src = "dedup d"
        src_key = "d.track_id"
    else:
        dedup_cte = f"""
        WITH dedup AS (
          SELECT DISTINCT dt, track_id
          FROM {SCHEMA}.stg_spotify__plays
          WHERE dt BETWEEN DATE '{s}' AND DATE '{e}'
        )
        """
        src = "dedup d"
        src_key = "d.track_id"

    # Try join to dim_tracks if present; else fallback
    try:
        # We won’t call information_schema for dim here; just try-then-fallback
        sql_join = f"""
        {dedup_cte}
        SELECT {src_key} AS track_id,
               COALESCE(t.track_name, {src_key}) AS track_name,
               COUNT(*) AS play_count
        FROM {src}
        LEFT JOIN {SCHEMA}.dim_tracks t
          ON {src_key} = t.track_id
        GROUP BY 1,2
        ORDER BY play_count DESC
        LIMIT {int(limit)}
        """
        df = read_sql(sql_join)
        if not df.empty:
            return df
    except Exception:
        pass

    sql_simple = f"""
    {dedup_cte}
    SELECT {src_key} AS track_id,
           {src_key} AS track_name,
           COUNT(*) AS play_count
    FROM {src}
    GROUP BY 1,2
    ORDER BY play_count DESC
    LIMIT {int(limit)}
    """
    return read_sql(sql_simple)


@st.cache_data(ttl=300)
def load_sessions(start_date, end_date):
    s, e = str(start_date), str(end_date)
    # Only query if fact table exists
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


# KPI tiles
kpis = load_kpis(start, end)
c1, c2, c3, c4 = st.columns(4)
if not kpis.empty:
    r = kpis.iloc[0].fillna(0)
    c1.metric("Total Plays", int(r["total_plays"] or 0))
    c2.metric("Unique Tracks", int(r["total_unique_tracks"] or 0))
    c3.metric("Avg Plays / Day", round(float(r["avg_plays_per_day"] or 0), 2))
    c4.metric("Active Days", int(r["active_days"] or 0))
    st.caption(f"Top track: **{r.get('top_track_name') or '—'}** ({int(r.get('top_track_plays') or 0)} plays)")
else:
    st.warning("No data for the selected range.")

st.divider()

# Trends
daily = load_daily_series(start, end)
left, right = st.columns([2,1])
with left:
    if not daily.empty:
        chart = alt.Chart(daily).mark_line(point=True).encode(
            x="dt:T", y="plays:Q", tooltip=["dt:T","plays:Q","unique_tracks:Q"]
        ).properties(height=280, title="Plays per day")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No daily data yet.")
with right:
    if not daily.empty:
        st.dataframe(daily.rename(columns={"dt":"Date","plays":"Plays","unique_tracks":"Unique tracks"}),
                     use_container_width=True, height=280)

st.divider()

# Top tracks
tops = load_top_tracks(start, end, limit=15)
if not tops.empty:
    bars = alt.Chart(tops).mark_bar().encode(
        x="play_count:Q", y=alt.Y("track_name:N", sort="-x"),
        tooltip=["track_name:N","play_count:Q"]
    ).properties(height=400, title="Top tracks")
    st.altair_chart(bars, use_container_width=True)
    st.dataframe(tops, use_container_width=True, hide_index=True)
else:
    st.info("No top tracks in this range.")

st.divider()

# Sessions
sessions = load_sessions(start, end)
st.subheader("Recent listening sessions (30-min gaps)")
if not sessions.empty:
    sessions = sessions.copy()
    sessions["duration_min"] = (sessions["session_duration_seconds"].astype(float) / 60).round(1)
    st.dataframe(
        sessions[["session_id","session_start","session_end","track_plays","duration_min"]],
        use_container_width=True, hide_index=True
    )
else:
    st.info("No sessions found in this range.")

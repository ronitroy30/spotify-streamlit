import os, datetime as dt
import pandas as pd
import streamlit as st
import altair as alt
from dateutil.relativedelta import relativedelta
from athena import read_sql

SCHEMA = st.secrets.get("ATHENA_SCHEMA", os.getenv("ATHENA_SCHEMA", "spotify_analytics"))

st.set_page_config(page_title="Spotify Analytics — dbt + Athena", layout="wide")
st.title("🎧 Spotify Analytics — dbt + Athena")

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
def load_kpis(start_date, end_date):
    sql = f"""
    with plays as (
      select cast(dt as date) as dt, track_id
      from {SCHEMA}.stg_spotify__plays
      where dt between :start and :end
    ),
    daily as (
      select dt, count(*) as plays, count(distinct track_id) as unique_tracks
      from plays group by 1
    ),
    totals as (
      select
        sum(plays) as total_plays,
        sum(unique_tracks) as total_unique_tracks,
        avg(plays) as avg_plays_per_day,
        count(*) as active_days
      from daily
    ),
    top_tracks as (
      select p.track_id, t.track_name, count(*) as play_count,
             row_number() over (order by count(*) desc) as rn
      from plays p
      left join {SCHEMA}.dim_tracks t using (track_id)
      group by 1,2
    )
    select
      total_plays,
      total_unique_tracks,
      avg_plays_per_day,
      active_days,
      (select track_name from top_tracks where rn = 1) as top_track_name,
      (select play_count from top_tracks where rn = 1) as top_track_plays
    from totals
    """
    return read_sql(sql, {"start": str(start_date), "end": str(end_date)})

@st.cache_data(ttl=300)
def load_daily_series(start_date, end_date):
    sql = f"""
    select cast(dt as date) as dt,
           count(*) as plays,
           count(distinct track_id) as unique_tracks
    from {SCHEMA}.stg_spotify__plays
    where dt between :start and :end
    group by 1
    order by 1
    """
    return read_sql(sql, {"start": str(start_date), "end": str(end_date)})

@st.cache_data(ttl=300)
def load_top_tracks(start_date, end_date, limit=15):
    sql = f"""
    select p.track_id, coalesce(t.track_name, p.track_id) as track_name, count(*) as play_count
    from {SCHEMA}.stg_spotify__plays p
    left join {SCHEMA}.dim_tracks t using (track_id)
    where p.dt between :start and :end
    group by 1,2
    order by play_count desc
    limit {int(limit)}
    """
    return read_sql(sql, {"start": str(start_date), "end": str(end_date)})

@st.cache_data(ttl=300)
def load_sessions(start_date, end_date):
    sql = f"""
    select session_id, session_start, session_end, track_plays, session_duration_seconds
    from {SCHEMA}.fct_listening_sessions
    where date(session_start) between :start and :end
    order by session_start desc
    limit 1000
    """
    return read_sql(sql, {"start": str(start_date), "end": str(end_date)})

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

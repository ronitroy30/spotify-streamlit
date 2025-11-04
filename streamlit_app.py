import os
import datetime as dt
import pandas as pd
import streamlit as st
import altair as alt
from dateutil.relativedelta import relativedelta
from athena import read_sql  # uses get_engine() inside athena.py

# ====== LIVE MODE (no AWS) — Spotify OAuth + on-the-fly analytics ======
import time, base64, urllib.parse, requests

def sp_auth_url():
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    redirect_uri = st.secrets["SPOTIFY_REDIRECT_URI"]
    scopes = st.secrets.get("SPOTIFY_SCOPES", "user-read-recently-played")
    state = str(int(time.time()))
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state,
        "show_dialog": "false",
    }
    return "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)

def sp_exchange_code_for_tokens(code: str):
    token_url = "https://accounts.spotify.com/api/token"
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    redirect_uri = st.secrets["SPOTIFY_REDIRECT_URI"]
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    auth = (client_id, client_secret)
    r = requests.post(token_url, data=data, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()  # access_token, refresh_token, expires_in, token_type

def sp_refresh_access_token(refresh_token: str):
    token_url = "https://accounts.spotify.com/api/token"
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    auth = (client_id, client_secret)
    r = requests.post(token_url, data=data, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()

def sp_get_headers(access_token: str):
    return {"Authorization": f"Bearer {access_token}"}

def sp_fetch_recently_played(access_token: str, limit=50, pages=10):
    # Pull up to ~500 recent plays
    items = []
    url = "https://api.spotify.com/v1/me/player/recently-played"
    headers = sp_get_headers(access_token)
    params = {"limit": min(limit, 50)}
    for _ in range(pages):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 401:
            raise RuntimeError("Unauthorized—token expired")
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("items", []))
        next_url = data.get("next")
        if not next_url:
            break
        # Spotify returns full URL for next page
        url = next_url
        params = {}
    return items

def normalize_recently_played(items: list) -> pd.DataFrame:
    rows = []
    for it in items:
        played_at = it.get("played_at")
        track = it.get("track", {}) or {}
        aid = (track.get("album") or {}).get("id")
        aname = (track.get("album") or {}).get("name")
        t_id = track.get("id")
        t_name = track.get("name")
        explicit = track.get("explicit")
        duration_ms = track.get("duration_ms")
        # Artists: take first for simplicity
        artists = track.get("artists") or []
        if artists:
            artist_id = artists[0].get("id")
            artist_name = artists[0].get("name")
        else:
            artist_id = None; artist_name = None
        rows.append({
            "played_at": played_at,
            "track_id": t_id,
            "track_name": t_name,
            "artist_id": artist_id,
            "artist_name": artist_name,
            "album_id": aid,
            "album_name": aname,
            "is_explicit": explicit,
            "duration_ms": duration_ms
        })
    df = pd.DataFrame(rows)
    if not df.empty and "played_at" in df.columns:
        df["played_at"] = pd.to_datetime(df["played_at"])
        df["dt"] = df["played_at"].dt.date
    return df

def live_mode_ui():
    st.subheader("🔌 Live mode (connect your Spotify)")
    query_params = st.query_params  # Streamlit Cloud supports this
    code = query_params.get("code", [None])[0] if isinstance(query_params.get("code"), list) else query_params.get("code")

    if "sp_access_token" not in st.session_state:
        st.session_state["sp_access_token"] = None
    if "sp_refresh_token" not in st.session_state:
        st.session_state["sp_refresh_token"] = None
    if "sp_token_expiry" not in st.session_state:
        st.session_state["sp_token_expiry"] = 0

    def need_refresh():
        return time.time() > st.session_state.get("sp_token_expiry", 0) - 60

    # Handle callback
    if code and not st.session_state["sp_access_token"]:
        try:
            tokens = sp_exchange_code_for_tokens(code)
            st.session_state["sp_access_token"] = tokens["access_token"]
            st.session_state["sp_refresh_token"] = tokens.get("refresh_token", st.session_state.get("sp_refresh_token"))
            st.session_state["sp_token_expiry"] = time.time() + int(tokens.get("expires_in", 3600))
            st.success("Spotify authorized!")
            # Clean the URL to remove the code param (nice UX)
            st.query_params.clear()
        except Exception as e:
            st.error("Failed to exchange Spotify code.")
            st.exception(e)

    col1, col2 = st.columns([1,1])
    with col1:
        if not st.session_state["sp_access_token"]:
            if st.button("Connect Spotify"):
                st.markdown(f"[Click here to authorize]({sp_auth_url()})")
        else:
            st.success("Connected to Spotify")

    with col2:
        if st.session_state["sp_access_token"]:
            if st.button("Disconnect"):
                for k in ["sp_access_token","sp_refresh_token","sp_token_expiry"]:
                    st.session_state.pop(k, None)
                st.experimental_rerun()

    # If connected: refresh if needed, fetch recent, and show a quick dashboard
    if st.session_state["sp_access_token"]:
        # refresh if needed
        if need_refresh() and st.session_state.get("sp_refresh_token"):
            try:
                rt = sp_refresh_access_token(st.session_state["sp_refresh_token"])
                st.session_state["sp_access_token"] = rt["access_token"]
                # Spotify may or may not return a new refresh_token
                if "refresh_token" in rt:
                    st.session_state["sp_refresh_token"] = rt["refresh_token"]
                st.session_state["sp_token_expiry"] = time.time() + int(rt.get("expires_in", 3600))
            except Exception as e:
                st.error("Token refresh failed; reconnect required.")
                st.exception(e)
                return

        try:
            with st.spinner("Fetching your recent plays…"):
                items = sp_fetch_recently_played(st.session_state["sp_access_token"], pages=10)
                df = normalize_recently_played(items)
        except Exception as e:
            st.error("Could not fetch recently played.")
            st.exception(e)
            return

        if df.empty:
            st.info("No recent plays returned by Spotify.")
            return

        # Date slicer using real timestamps
        left, right = st.columns(2)
        min_d = df["played_at"].min().date()
        max_d = df["played_at"].max().date()
        with left:
            s = st.date_input("Start (live)", value=min_d, min_value=min_d, max_value=max_d, key="live_start")
        with right:
            e = st.date_input("End (live)", value=max_d, min_value=min_d, max_value=max_d, key="live_end")

        sdf = df[(df["played_at"].dt.date >= s) & (df["played_at"].dt.date <= e)].copy()
        # Dedupe by (played_at, track_id)
        sdf = sdf.drop_duplicates(subset=["played_at","track_id"], keep="first")

        # KPIs
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Plays", len(sdf))
        k2.metric("Unique Tracks", sdf["track_id"].nunique())
        by_day = sdf.groupby(sdf["played_at"].dt.date).size()
        k3.metric("Avg Plays / Day", round(by_day.mean() if not by_day.empty else 0, 2))
        k4.metric("Active Days", int(by_day.size))

        # Daily chart
        trend = by_day.reset_index()
        trend.columns = ["dt","plays"]
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x=alt.X("dt:T", title="Date"),
            y=alt.Y("plays:Q", title="Plays"),
            tooltip=["dt:T","plays:Q"]
        ).properties(height=260, title="Plays per day (live)")
        st.altair_chart(chart, use_container_width=True)

        # Top tracks (name in response)
        tops = sdf.groupby(["track_id","track_name"], dropna=False).size().reset_index(name="play_count")
        tops = tops.sort_values("play_count", ascending=False).head(15)
        bars = alt.Chart(tops).mark_bar().encode(
            x=alt.X("play_count:Q", title="Plays"),
            y=alt.Y("track_name:N", sort="-x", title="Track"),
            tooltip=["track_name:N","play_count:Q"]
        ).properties(height=360, title="Top tracks (live)")
        st.altair_chart(bars, use_container_width=True)
        st.dataframe(tops, use_container_width=True, hide_index=True)

# ====== Toggle between your AWS-backed dashboard and Live mode ======
st.divider()
mode = st.radio("Mode", ["Demo mode", "Live mode (user connects Spotify)"], horizontal=True)
if mode == "Live mode (user connects Spotify)":
    live_mode_ui()
    st.stop()


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

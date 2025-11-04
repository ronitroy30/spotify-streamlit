import os
import time
import itertools
import urllib.parse
import datetime as dt
import pandas as pd
import requests
import streamlit as st
import altair as alt
from dateutil.relativedelta import relativedelta

# ----------------------------- Config -----------------------------
st.set_page_config(page_title="Spotify Analytics — Demo + Live", layout="wide")
SCHEMA = st.secrets.get("ATHENA_SCHEMA", os.getenv("ATHENA_SCHEMA", "spotify_analytics"))

# Optional Athena helper (demo mode). App still runs if missing.
try:
    from athena import read_sql  # def read_sql(sql, params=None)->pd.DataFrame
    ATHENA_AVAILABLE = True
except Exception:
    ATHENA_AVAILABLE = False

# ----------------------- Spotify OAuth helpers --------------------
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
    auth = (st.secrets["SPOTIFY_CLIENT_ID"], st.secrets["SPOTIFY_CLIENT_SECRET"])
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": st.secrets["SPOTIFY_REDIRECT_URI"]}
    r = requests.post(token_url, data=data, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()

def sp_refresh_access_token(refresh_token: str):
    token_url = "https://accounts.spotify.com/api/token"
    auth = (st.secrets["SPOTIFY_CLIENT_ID"], st.secrets["SPOTIFY_CLIENT_SECRET"])
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    r = requests.post(token_url, data=data, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()

def sp_get_headers(access_token: str):
    return {"Authorization": f"Bearer {access_token}"}

def sp_fetch_recently_played(access_token: str, limit=50, pages=10):
    items = []
    url = "https://api.spotify.com/v1/me/player/recently-played"
    headers = sp_get_headers(access_token)
    params = {"limit": min(limit, 50)}
    for _ in range(max(1, pages)):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code in (401, 403):
            try:
                err = r.json()
            except Exception:
                err = {"error": {"status": r.status_code, "message": r.text}}
            raise RuntimeError(f"Spotify API {r.status_code}: {err}")
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("items", []))
        next_url = data.get("next")
        if not next_url:
            break
        url = next_url
        params = {}
    return items

def normalize_recently_played(items: list) -> pd.DataFrame:
    rows = []
    for it in items:
        played_at = it.get("played_at")
        track = it.get("track") or {}
        album = track.get("album") or {}
        artists = track.get("artists") or []
        artist_id = artists[0].get("id") if artists else None
        artist_name = artists[0].get("name") if artists else None
        rows.append({
            "played_at": played_at,
            "track_id": track.get("id"),
            "track_name": track.get("name"),
            "artist_id": artist_id,
            "artist_name": artist_name,
            "album_id": album.get("id"),
            "album_name": album.get("name"),
            "is_explicit": track.get("explicit"),
            "duration_ms": track.get("duration_ms"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "played_at" in df.columns:
        df["played_at"] = pd.to_datetime(df["played_at"])
        df["date"] = df["played_at"].dt.date
        df["hour"] = df["played_at"].dt.hour
        df["weekday"] = df["played_at"].dt.day_name()
    return df

# ------------------- Live helpers: sessions & enrich ---------------
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def make_sessions(sdf, gap_minutes=30):
    if sdf.empty:
        return pd.DataFrame(columns=["session_id","session_start","session_end","track_plays","session_duration_seconds"])
    sdf = sdf.sort_values("played_at").copy()
    gap = pd.Timedelta(minutes=gap_minutes)
    new_session = (sdf["played_at"].diff() > gap) | (sdf["played_at"].diff().isna())
    sdf["session_id"] = new_session.cumsum()
    ag = sdf.groupby("session_id").agg(
        session_start=("played_at","min"),
        session_end=("played_at","max"),
        track_plays=("track_id","count"),
    ).reset_index()
    ag["session_duration_seconds"] = (ag["session_end"] - ag["session_start"]).dt.total_seconds().astype(int)
    return ag

def sp_audio_features(access_token, track_ids):
    if not track_ids:
        return pd.DataFrame()
    headers = {"Authorization": f"Bearer {access_token}"}
    rows = []
    for batch in chunked(list(dict.fromkeys(track_ids)), 100):
        r = requests.get("https://api.spotify.com/v1/audio-features",
                         headers=headers, params={"ids": ",".join(batch)}, timeout=30)
        if r.status_code in (401, 403):
            break
        r.raise_for_status()
        for f in (r.json().get("audio_features") or []):
            if not f:
                continue
            rows.append({
                "track_id": f.get("id"),
                "danceability": f.get("danceability"),
                "energy": f.get("energy"),
                "valence": f.get("valence"),
                "tempo": f.get("tempo"),
                "acousticness": f.get("acousticness"),
                "instrumentalness": f.get("instrumentalness"),
                "liveness": f.get("liveness"),
                "speechiness": f.get("speechiness"),
            })
    return pd.DataFrame(rows)

def sp_artists_genres(access_token, artist_ids):
    if not artist_ids:
        return pd.DataFrame(columns=["artist_id","genre"])
    headers = {"Authorization": f"Bearer {access_token}"}
    rows = []
    for batch in chunked(list(dict.fromkeys(artist_ids)), 50):
        r = requests.get("https://api.spotify.com/v1/artists",
                         headers=headers, params={"ids": ",".join(batch)}, timeout=30)
        if r.status_code in (401,403):
            break
        r.raise_for_status()
        for a in r.json().get("artists", []):
            for g in (a.get("genres") or []):
                rows.append({"artist_id": a.get("id"), "genre": g})
    return pd.DataFrame(rows)

# =============================== UI ===============================
st.title("🎧 Spotify Analytics")

mode = st.radio("Mode", ["Demo mode", "Live mode (connect Spotify)"], horizontal=True)

# ---------------------------- Demo mode ---------------------------
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
    if "played_at" in cols: return "played_at"
    if "played_at_ts" in cols: return "played_at_ts"
    return "dt"

# Cache loaders with explicit keys that include date range (as strings)
@st.cache_data(ttl=300, show_spinner=False)
def load_kpis_demo(schema: str, start: str, end: str):
    cols = get_table_columns(schema, "stg_spotify__plays")
    tcol = plays_time_expr(cols)
    if tcol in ("played_at", "played_at_ts"):
        sql = f"""
        WITH base AS (
          SELECT {tcol} AS ts, track_id
          FROM {schema}.stg_spotify__plays
          WHERE DATE({tcol}) BETWEEN DATE '{start}' AND DATE '{end}'
        ),
        daily AS (
          SELECT CAST(date(ts) AS date) AS dt,
                 COUNT(*) AS plays,
                 COUNT(DISTINCT track_id) AS unique_tracks
          FROM base GROUP BY 1
        )
        SELECT
          SUM(plays) AS total_plays,
          SUM(unique_tracks) AS total_unique_tracks,
          AVG(plays) AS avg_plays_per_day,
          COUNT(*) AS active_days
        FROM daily
        """
    else:
        sql = f"""
        WITH base AS (
          SELECT dt, track_id
          FROM {schema}.stg_spotify__plays
          WHERE dt BETWEEN DATE '{start}' AND DATE '{end}'
        ),
        daily AS (
          SELECT CAST(dt AS date) AS dt,
                 COUNT(*) AS plays,
                 COUNT(DISTINCT track_id) AS unique_tracks
          FROM base GROUP BY 1
        )
        SELECT
          SUM(plays) AS total_plays,
          SUM(unique_tracks) AS total_unique_tracks,
          AVG(plays) AS avg_plays_per_day,
          COUNT(*) AS active_days
        FROM daily
        """
    return read_sql(sql)

@st.cache_data(ttl=300, show_spinner=False)
def load_daily_demo(schema: str, start: str, end: str):
    cols = get_table_columns(schema, "stg_spotify__plays")
    tcol = plays_time_expr(cols)
    if tcol in ("played_at", "played_at_ts"):
        sql = f"""
        SELECT CAST(date({tcol}) AS date) AS dt,
               COUNT(*) AS plays,
               COUNT(DISTINCT track_id) AS unique_tracks
        FROM {schema}.stg_spotify__plays
        WHERE DATE({tcol}) BETWEEN DATE '{start}' AND DATE '{end}'
        GROUP BY 1
        ORDER BY 1
        """
    else:
        sql = f"""
        SELECT CAST(dt AS date) AS dt,
               COUNT(*) AS plays,
               COUNT(DISTINCT track_id) AS unique_tracks
        FROM {schema}.stg_spotify__plays
        WHERE dt BETWEEN DATE '{start}' AND DATE '{end}'
        GROUP BY 1
        ORDER BY 1
        """
    return read_sql(sql)

@st.cache_data(ttl=300, show_spinner=False)
def load_top_tracks_demo(schema: str, start: str, end: str, limit: int):
    cols = get_table_columns(schema, "stg_spotify__plays")
    tcol = plays_time_expr(cols)
    time_filter = (
        f"DATE({tcol}) BETWEEN DATE '{start}' AND DATE '{end}'"
        if tcol in ("played_at", "played_at_ts")
        else f"dt BETWEEN DATE '{start}' AND DATE '{end}'"
    )
    # prefer joining to dim_tracks for names
    try:
        sql = f"""
        SELECT p.track_id,
               COALESCE(t.track_name, p.track_id) AS track_name,
               COUNT(*) AS play_count
        FROM {schema}.stg_spotify__plays p
        LEFT JOIN {schema}.dim_tracks t ON p.track_id = t.track_id
        WHERE {time_filter}
        GROUP BY 1,2
        ORDER BY play_count DESC
        LIMIT {int(limit)}
        """
        df = read_sql(sql)
        if not df.empty:
            return df
    except Exception:
        pass
    sql = f"""
    SELECT track_id,
           track_id AS track_name,
           COUNT(*) AS play_count
    FROM {schema}.stg_spotify__plays
    WHERE {time_filter}
    GROUP BY 1,2
    ORDER BY play_count DESC
    LIMIT {int(limit)}
    """
    return read_sql(sql)

@st.cache_data(ttl=300, show_spinner=False)
def load_sessions_demo(schema: str, start: str, end: str):
    q_exists = f"""
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = '{schema}'
      AND table_name   = 'fct_listening_sessions'
    LIMIT 1
    """
    ex = read_sql(q_exists)
    if ex.empty:
        return pd.DataFrame(columns=["session_id","session_start","session_end","track_plays","session_duration_seconds"])
    sql = f"""
    SELECT session_id, session_start, session_end, track_plays, session_duration_seconds
    FROM {schema}.fct_listening_sessions
    WHERE DATE(session_start) BETWEEN DATE '{start}' AND DATE '{end}'
    ORDER BY session_start DESC
    LIMIT 1000
    """
    return read_sql(sql)

def demo_mode_ui():
    st.subheader("🧪 Demo mode")
    if not ATHENA_AVAILABLE:
        st.warning("Demo mode unavailable: `athena.py` not found or import failed.")
        return

    # Single date range (this is the only one shown in Demo mode)
    default_end = dt.date.today()
    default_start = default_end - relativedelta(months=1)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        start = st.date_input("Start date", value=default_start, key="demo_start")
    with col_b:
        end = st.date_input("End date", value=default_end, key="demo_end")

    # Normalize to strings for caching + SQL
    s_str, e_str = str(start), str(end)

    # KPIs
    try:
        kpis = load_kpis_demo(SCHEMA, s_str, e_str)
        c1,c2,c3,c4 = st.columns(4)
        if not kpis.empty:
            r = kpis.iloc[0].fillna(0)
            c1.metric("Total Plays", int(r.get("total_plays", 0) or 0))
            c2.metric("Unique Tracks", int(r.get("total_unique_tracks", 0) or 0))
            c3.metric("Avg Plays / Day", round(float(r.get("avg_plays_per_day", 0) or 0), 2))
            c4.metric("Active Days", int(r.get("active_days", 0) or 0))
        else:
            st.info("No data for the selected range.")
    except Exception as e:
        st.error("Failed to load KPIs.")
        st.exception(e)

    st.divider()

    # Trends
    try:
        daily = load_daily_demo(SCHEMA, s_str, e_str)
        left, right = st.columns([2, 1])
        with left:
            if not daily.empty:
                chart = alt.Chart(daily).mark_line(point=True).encode(
                    x=alt.X("dt:T", title="Date"),
                    y=alt.Y("plays:Q", title="Plays"),
                    tooltip=["dt:T", "plays:Q", "unique_tracks:Q"]
                ).properties(height=280, title="Plays per day")
                st.altair_chart(chart, use_container_width=True)

                daily = daily.sort_values("dt").copy()
                daily["cum_plays"] = daily["plays"].cumsum()
                st.altair_chart(
                    alt.Chart(daily).mark_line(point=True).encode(
                        x="dt:T", y=alt.Y("cum_plays:Q", title="Cumulative plays"),
                        tooltip=["dt:T","cum_plays:Q"]
                    ).properties(height=220, title="Cumulative plays"),
                    use_container_width=True
                )
            else:
                st.info("No daily data yet.")
        with right:
            if not daily.empty:
                st.dataframe(
                    daily.rename(columns={"dt":"Date","plays":"Plays","unique_tracks":"Unique tracks"}),
                    use_container_width=True, height=280
                )
    except Exception as e:
        st.error("Failed to load daily series.")
        st.exception(e)

    st.divider()

    # Top tracks
    try:
        tops = load_top_tracks_demo(SCHEMA, s_str, e_str, limit=15)
        if not tops.empty:
            st.altair_chart(
                alt.Chart(tops).mark_bar().encode(
                    x=alt.X("play_count:Q", title="Plays"),
                    y=alt.Y("track_name:N", sort="-x", title="Track"),
                    tooltip=["track_name:N","play_count:Q"]
                ).properties(height=400, title="Top tracks"),
                use_container_width=True
            )
            st.dataframe(tops, use_container_width=True, hide_index=True)
        else:
            st.info("No top tracks in this range.")
    except Exception as e:
        st.error("Failed to load top tracks.")
        st.exception(e)

    st.divider()

    # Sessions
    st.subheader("Recent listening sessions (30-min gaps)")
    try:
        sessions = load_sessions_demo(SCHEMA, s_str, e_str)
        if not sessions.empty:
            sessions = sessions.copy()
            sessions["duration_min"] = (sessions["session_duration_seconds"].astype(float)/60).round(1)
            st.dataframe(
                sessions[["session_id","session_start","session_end","track_plays","duration_min"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No sessions found in this range.")
    except Exception as e:
        st.error("Failed to load sessions.")
        st.exception(e)

# ---------------------------- Live mode ---------------------------
def fmt_pct(x):
    try: return f"{x:+.0%}"
    except Exception: return "—"

def period_compare(df, start, end, prev_mode="previous"):
    df = df.copy()
    df["date"] = df["played_at"].dt.date
    curr = df[(df["date"] >= start) & (df["date"] <= end)]
    if prev_mode == "previous":
        days = (end - start).days + 1
        base_start = start - pd.Timedelta(days=days)
        base_end   = start - pd.Timedelta(days=1)
        base_label = f"Prev {days}d ({base_start} → {base_end})"
    else:
        ps, pe = prev_mode
        base_start, base_end = ps, pe
        base_label = f"Baseline ({ps} → {pe})"
    base = df[(df["date"] >= base_start) & (df["date"] <= base_end)]
    return curr, base, base_label

def kpi_block(col, label, value, delta=None):
    if delta is None: col.metric(label, value)
    else: col.metric(label, value, delta=fmt_pct(delta))

def live_mode_ui():
    st.subheader("🔌 Live mode (connect your Spotify)")
    # One (and only one) date range picker inside Live mode
    qp = st.query_params
    code = qp.get("code", [None])[0] if isinstance(qp.get("code"), list) else qp.get("code")

    if "sp_access_token" not in st.session_state:
        st.session_state.update(sp_access_token=None, sp_refresh_token=None, sp_token_expiry=0, sp_scope="")

    def need_refresh():
        return time.time() > st.session_state.get("sp_token_expiry", 0) - 60

    if code and not st.session_state["sp_access_token"]:
        try:
            tokens = sp_exchange_code_for_tokens(code)
            st.session_state["sp_access_token"] = tokens["access_token"]
            st.session_state["sp_refresh_token"] = tokens.get("refresh_token", st.session_state.get("sp_refresh_token"))
            st.session_state["sp_token_expiry"] = time.time() + int(tokens.get("expires_in", 3600))
            st.session_state["sp_scope"] = tokens.get("scope","")
            st.success("Spotify authorized!")
            st.caption(f"Granted scopes: `{st.session_state['sp_scope']}`")
            st.query_params.clear()
        except Exception as e:
            st.error("Failed to exchange Spotify code.")
            st.exception(e)

    c1,c2 = st.columns([1,1])
    with c1:
        if not st.session_state["sp_access_token"]:
            if st.button("Connect Spotify"):
                st.markdown(f"[Click here to authorize]({sp_auth_url()})")
        else:
            st.success("Connected to Spotify")
            if st.session_state.get("sp_scope"):
                st.caption(f"Scopes: `{st.session_state['sp_scope']}`")
    with c2:
        if st.session_state["sp_access_token"]:
            if st.button("Disconnect"):
                for k in ["sp_access_token","sp_refresh_token","sp_token_expiry","sp_scope"]:
                    st.session_state.pop(k, None)
                st.experimental_rerun()

    if not st.session_state["sp_access_token"]:
        st.info("Connect Spotify to see your live analytics.")
        return

    if need_refresh() and st.session_state.get("sp_refresh_token"):
        try:
            rt = sp_refresh_access_token(st.session_state["sp_refresh_token"])
            st.session_state["sp_access_token"] = rt["access_token"]
            if "refresh_token" in rt:
                st.session_state["sp_refresh_token"] = rt["refresh_token"]
            st.session_state["sp_token_expiry"] = time.time() + int(rt.get("expires_in", 3600))
            if "scope" in rt:
                st.session_state["sp_scope"] = rt["scope"]
        except Exception as e:
            st.error("Token refresh failed; reconnect required.")
            st.exception(e)
            return

    # Fetch once, then filter by the single date range
    try:
        with st.spinner("Fetching your recent plays…"):
            items = sp_fetch_recently_played(st.session_state["sp_access_token"], pages=10)
            df = normalize_recently_played(items)
    except Exception as e:
        st.error("Could not fetch recently played.")
        st.exception(e)
        st.info("Ensure your account is a tester (Dev mode) and scope `user-read-recently-played` is granted.")
        return

    if df.empty:
        st.info("No recent plays returned by Spotify. Play something and refresh.")
        return

    # Single date range for Live mode
    min_d, max_d = df["date"].min(), df["date"].max()
    cA, cB = st.columns([1,1])
    with cA:
        s = st.date_input("Start", value=min_d, min_value=min_d, max_value=max_d, key="live_s")
    with cB:
        e = st.date_input("End", value=max_d, min_value=min_d, max_value=max_d, key="live_e")

    # Filter to range
    curr = df[(df["date"] >= s) & (df["date"] <= e)].copy()
    if curr.empty:
        st.info("No plays in the selected range.")
        return

    # KPIs
    by_day = curr.groupby("date").size()
    total = len(curr)
    uniq = curr["track_id"].nunique()
    avg = by_day.mean() if not by_day.empty else 0
    days = by_day.size
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Plays", total)
    k2.metric("Unique Tracks", uniq)
    k3.metric("Avg Plays / Day", round(avg,2))
    k4.metric("Active Days", days)

    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trend", "Leaders", "Sessions", "Heatmaps", "Audio & Genres"])

    # Trend
    with tab1:
        trend = curr.groupby("date").size().reset_index(name="plays").sort_values("date")
        trend["roll7"] = trend["plays"].rolling(7, min_periods=1).mean()
        tm = trend.melt(id_vars=["date"], value_vars=["plays","roll7"], var_name="series", value_name="value")
        st.altair_chart(
            alt.Chart(tm).mark_line(point=True).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Plays"),
                color=alt.Color("series:N", title=""),
                tooltip=["date:T","series:N","value:Q"]
            ).properties(height=300, title="Daily plays (7-day rolling)"),
            use_container_width=True
        )
        trend["cum_plays"] = trend["plays"].cumsum()
        st.altair_chart(
            alt.Chart(trend).mark_line(point=True).encode(
                x="date:T",
                y=alt.Y("cum_plays:Q", title="Cumulative plays"),
                tooltip=["date:T","cum_plays:Q"]
            ).properties(height=220, title="Cumulative plays"),
            use_container_width=True
        )

    # Leaders
    with tab2:
        lt, la = st.columns(2)
        top_tracks = curr.groupby(["track_id","track_name"], dropna=False).size().reset_index(name="play_count").sort_values("play_count", ascending=False).head(15)
        top_artists = curr.groupby(["artist_id","artist_name"], dropna=False).size().reset_index(name="play_count").sort_values("play_count", ascending=False).head(15)
        top_albums  = curr.groupby(["album_id","album_name"], dropna=False).size().reset_index(name="play_count").sort_values("play_count", ascending=False).head(15)

        with lt:
            st.altair_chart(
                alt.Chart(top_tracks).mark_bar().encode(
                    x=alt.X("play_count:Q", title="Plays"),
                    y=alt.Y("track_name:N", sort="-x", title="Track"),
                    tooltip=["track_name:N","play_count:Q"]
                ).properties(height=380, title="Top tracks"),
                use_container_width=True
            )
        with la:
            st.altair_chart(
                alt.Chart(top_artists).mark_bar().encode(
                    x=alt.X("play_count:Q", title="Plays"),
                    y=alt.Y("artist_name:N", sort="-x", title="Artist"),
                    tooltip=["artist_name:N","play_count:Q"]
                ).properties(height=380, title="Top artists"),
                use_container_width=True
            )
        st.altair_chart(
            alt.Chart(top_albums).mark_bar().encode(
                x=alt.X("play_count:Q", title="Plays"),
                y=alt.Y("album_name:N", sort="-x", title="Album"),
                tooltip=["album_name:N","play_count:Q"]
            ).properties(height=320, title="Top albums"),
            use_container_width=True
        )

        # Diversity curve (no expanding.apply on strings)
        curr_sorted = curr.sort_values("played_at").copy()
        s_tracks = curr_sorted["track_id"].astype("string").fillna("__NULL__")
        first_time = ~s_tracks.duplicated()
        curr_sorted["cum_unique_tracks"] = first_time.cumsum()
        cum = curr_sorted[["played_at","cum_unique_tracks"]]
        st.altair_chart(
            alt.Chart(cum).mark_line().encode(
                x=alt.X("played_at:T", title="Time"),
                y=alt.Y("cum_unique_tracks:Q", title="Cumulative unique tracks"),
                tooltip=["played_at:T","cum_unique_tracks:Q"]
            ).properties(height=220, title="Discovery / diversity curve"),
            use_container_width=True
        )

    # Sessions
    with tab3:
        gap = st.slider("Session gap (minutes)", 10, 90, 30, step=5)
        sessions = make_sessions(curr, gap_minutes=gap)
        if sessions.empty:
            st.info("No sessions in this range.")
        else:
            cA,cB,cC = st.columns(3)
            cA.metric("Sessions", len(sessions))
            cB.metric("Avg duration (min)", round(sessions["session_duration_seconds"].mean()/60, 1))
            cC.metric("Max tracks in a session", int(sessions["track_plays"].max()))
            st.dataframe(sessions, use_container_width=True, hide_index=True)

    # Heatmaps
    with tab4:
        heat = curr.groupby(["weekday","hour"]).size().reset_index(name="plays")
        weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat["weekday"] = pd.Categorical(heat["weekday"], categories=weekday_order, ordered=True)
        st.altair_chart(
            alt.Chart(heat).mark_rect().encode(
                x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("weekday:O", title="Weekday", sort=weekday_order),
                color=alt.Color("plays:Q", title="Plays"),
                tooltip=["weekday:O","hour:O","plays:Q"]
            ).properties(height=260, title="Listening intensity by weekday & hour"),
            use_container_width=True
        )

    # Audio & Genres (optional extra calls)
    with tab5:
        enrich = st.checkbox("Enrich with audio features & genres (slower, extra API calls)", value=False)
        if enrich:
            with st.spinner("Fetching audio features…"):
                feats = sp_audio_features(st.session_state["sp_access_token"], curr["track_id"].dropna().unique().tolist())
            if not feats.empty:
                merged = curr.merge(feats, on="track_id", how="left")
                st.altair_chart(
                    alt.Chart(merged.dropna(subset=["danceability","energy"])).mark_circle(size=60).encode(
                        x=alt.X("danceability:Q", title="Danceability"),
                        y=alt.Y("energy:Q", title="Energy"),
                        tooltip=["track_name:N","artist_name:N","danceability:Q","energy:Q"]
                    ).properties(height=300, title="Danceability vs Energy"),
                    use_container_width=True
                )
            with st.spinner("Fetching artist genres…"):
                gdf = sp_artists_genres(st.session_state["sp_access_token"], curr["artist_id"].dropna().unique().tolist())
            if not gdf.empty:
                top_genres = gdf["genre"].value_counts().reset_index()
                top_genres.columns = ["genre","count"]
                st.altair_chart(
                    alt.Chart(top_genres.head(20)).mark_bar().encode(
                        x=alt.X("count:Q", title="Count"),
                        y=alt.Y("genre:N", sort="-x", title="Genre"),
                        tooltip=["genre:N","count:Q"]
                    ).properties(height=400, title="Top genres"),
                    use_container_width=True
                )

# ----------------------------- Router -----------------------------
if mode == "Demo mode":
    demo_mode_ui()
else:
    live_mode_ui()

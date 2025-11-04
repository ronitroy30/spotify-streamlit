import os
import time
import itertools
import urllib.parse
import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from dateutil.relativedelta import relativedelta

# ---- If you're using the Athena warehouse mode, keep athena.py with read_sql() ----
# athena.py should expose: def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame
try:
    from athena import read_sql  # noqa
    ATHENA_AVAILABLE = True
except Exception:
    ATHENA_AVAILABLE = False

# ---------- App config ----------
SCHEMA = st.secrets.get("ATHENA_SCHEMA", os.getenv("ATHENA_SCHEMA", "spotify_analytics"))
st.set_page_config(page_title="Spotify Analytics — dbt + Athena + Live", layout="wide")

st.title("🎧 Spotify Analytics — Warehouse + Live (Spotify OAuth)")

# ============================================================================
#                               LIVE MODE HELPERS
# ============================================================================

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
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": st.secrets["SPOTIFY_REDIRECT_URI"],
    }
    r = requests.post(token_url, data=data, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()  # {access_token, refresh_token, scope, expires_in, ...}

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
    """Paginate; surface Spotify 401/403 payloads in errors."""
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
        params = {}  # next includes cursors/params
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
        artists = track.get("artists") or []
        if artists:
            artist_id = artists[0].get("id")
            artist_name = artists[0].get("name")
        else:
            artist_id = None
            artist_name = None
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

# ---- Live utilities: comparisons, sessions, audio features, genres ----

def fmt_pct(x):
    try:
        return f"{x:+.0%}"
    except Exception:
        return "—"

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
    if delta is None:
        col.metric(label, value)
    else:
        col.metric(label, value, delta=fmt_pct(delta))

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
        track_plays=("track_id","count")
    ).reset_index()
    ag["session_duration_seconds"] = (ag["session_end"] - ag["session_start"]).dt.total_seconds().astype(int)
    return ag

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def sp_audio_features(access_token, track_ids):
    if not track_ids:
        return pd.DataFrame()
    headers = {"Authorization": f"Bearer {access_token}"}
    rows = []
    for batch in chunked(list(dict.fromkeys(track_ids)), 100):
        r = requests.get("https://api.spotify.com/v1/audio-features",
                         headers=headers, params={"ids": ",".join(batch)}, timeout=30)
        if r.status_code in (401,403):
            break
        r.raise_for_status()
        data = r.json().get("audio_features") or []
        for f in data:
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
                "speechiness": f.get("speechiness")
            })
    return pd.DataFrame(rows)

def sp_artists_genres(access_token, artist_ids):
    if not artist_ids:
        return pd.DataFrame(columns=["artist_id","genre"])
    headers = {"Authorization": f"Bearer {access_token}"}
    genre_rows = []
    for batch in chunked(list(dict.fromkeys(artist_ids)), 50):
        r = requests.get("https://api.spotify.com/v1/artists",
                         headers=headers, params={"ids": ",".join(batch)}, timeout=30)
        if r.status_code in (401,403):
            break
        r.raise_for_status()
        for a in r.json().get("artists", []):
            aid = a.get("id")
            for g in (a.get("genres") or []):
                genre_rows.append({"artist_id": aid, "genre": g})
    return pd.DataFrame(genre_rows)

def live_mode_ui():
    st.subheader("🔌 Live mode (connect your Spotify)")
    query_params = st.query_params
    code = query_params.get("code", [None])[0] if isinstance(query_params.get("code"), list) else query_params.get("code")

    if "sp_access_token" not in st.session_state:
        st.session_state.update(sp_access_token=None, sp_refresh_token=None, sp_token_expiry=0, sp_scope="")

    def need_refresh():
        return time.time() > st.session_state.get("sp_token_expiry", 0) - 60

    # Handle callback
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

    c1, c2 = st.columns([1,1])
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

    # Refresh token if near expiry
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

    # Fetch & normalize
    try:
        with st.spinner("Fetching your recent plays…"):
            items = sp_fetch_recently_played(st.session_state["sp_access_token"], pages=10)
            df = normalize_recently_played(items)
    except Exception as e:
        st.error("Could not fetch recently played.")
        st.exception(e)
        st.info("Common fixes: ensure your Spotify account is added as a tester or your app is approved; scope `user-read-recently-played` must be granted.")
        return

    if df.empty:
        st.info("No recent plays returned by Spotify. Play a few tracks and refresh.")
        return

    # Deduplicate & enrich
    df = df.drop_duplicates(subset=["played_at","track_id"], keep="first")
    df["date"] = df["played_at"].dt.date
    df["hour"] = df["played_at"].dt.hour
    df["weekday"] = df["played_at"].dt.day_name()

    # Date/compare controls
    min_d, max_d = df["date"].min(), df["date"].max()
    cA, cB, cC = st.columns([1,1,1])
    with cA:
        s = st.date_input("Start", value=min_d, min_value=min_d, max_value=max_d, key="live_s")
    with cB:
        e = st.date_input("End", value=max_d, min_value=min_d, max_value=max_d, key="live_e")
    with cC:
        compare_mode = st.selectbox("Compare to", ["None", "Previous period", "Custom range"], index=1)

    base_mode = None
    if compare_mode == "Previous period":
        base_mode = "previous"
    elif compare_mode == "Custom range":
        bc1, bc2 = st.columns(2)
        with bc1:
            bs = st.date_input("Baseline start", value=s - pd.Timedelta(days=(e - s).days + 1), key="base_s")
        with bc2:
            be = st.date_input("Baseline end", value=s - pd.Timedelta(days=1), key="base_e")
        base_mode = (bs, be)

    curr, base, base_label = period_compare(df, s, e, prev_mode=base_mode if base_mode else "previous") if compare_mode != "None" else (df[(df["date"]>=s)&(df["date"]<=e)], pd.DataFrame(), "")

    def kpi_stats(x):
        total = len(x)
        uniq = x["track_id"].nunique()
        by_day = x.groupby("date").size()
        avg = by_day.mean() if not by_day.empty else 0
        days = by_day.size
        return total, uniq, avg, days

    t, u, a, d = kpi_stats(curr)
    bt=bu=ba=bd=0
    if not base.empty:
        bt, bu, ba, bd = kpi_stats(base)

    k1,k2,k3,k4 = st.columns(4)
    if base.empty:
        kpi_block(k1, "Total Plays", t)
        kpi_block(k2, "Unique Tracks", u)
        kpi_block(k3, "Avg Plays / Day", round(a,2))
        kpi_block(k4, "Active Days", d)
    else:
        kpi_block(k1, "Total Plays", t, (t - bt)/bt if bt else None)
        kpi_block(k2, "Unique Tracks", u, (u - bu)/bu if bu else None)
        kpi_block(k3, "Avg Plays / Day", round(a,2), (a - ba)/ba if ba else None)
        kpi_block(k4, "Active Days", d, (d - bd)/bd if bd else None)
        st.caption(f"Baseline: {base_label}")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trend", "Leaders", "Sessions", "Heatmaps", "Audio & Genres"])

    # Trend
    with tab1:
        trend = curr.groupby("date").size().reset_index(name="plays").sort_values("date")
        if not trend.empty:
            trend["roll7"] = trend["plays"].rolling(7, min_periods=1).mean()
            trend_melt = trend.melt(id_vars=["date"], value_vars=["plays","roll7"], var_name="series", value_name="value")
            ch = alt.Chart(trend_melt).mark_line(point=True).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Plays"),
                color=alt.Color("series:N", title=""),
                tooltip=["date:T","series:N","value:Q"]
            ).properties(height=300, title="Daily plays (with 7-day rolling avg)")
            st.altair_chart(ch, use_container_width=True)

            trend["cum_plays"] = trend["plays"].cumsum()
            st.altair_chart(
                alt.Chart(trend).mark_line(point=True).encode(
                    x="date:T", y=alt.Y("cum_plays:Q", title="Cumulative plays"),
                    tooltip=["date:T","cum_plays:Q"]
                ).properties(height=220, title="Cumulative plays"),
                use_container_width=True
            )
        else:
            st.info("No plays in the selected range.")

    # Leaders
    with tab2:
        lt, la = st.columns(2)
        top_tracks = curr.groupby(["track_id","track_name"], dropna=False).size().reset_index(name="play_count").sort_values("play_count", ascending=False).head(15)
        top_artists = curr.groupby(["artist_id","artist_name"], dropna=False).size().reset_index(name="play_count").sort_values("play_count", ascending=False).head(15)
        top_albums = curr.groupby(["album_id","album_name"], dropna=False).size().reset_index(name="play_count").sort_values("play_count", ascending=False).head(15)

        with lt:
            st.altair_chart(alt.Chart(top_tracks).mark_bar().encode(
                x=alt.X("play_count:Q", title="Plays"),
                y=alt.Y("track_name:N", sort="-x", title="Track"),
                tooltip=["track_name:N","play_count:Q"]
            ).properties(height=380, title="Top tracks"), use_container_width=True)
        with la:
            st.altair_chart(alt.Chart(top_artists).mark_bar().encode(
                x=alt.X("play_count:Q", title="Plays"),
                y=alt.Y("artist_name:N", sort="-x", title="Artist"),
                tooltip=["artist_name:N","play_count:Q"]
            ).properties(height=380, title="Top artists"), use_container_width=True)

        st.altair_chart(alt.Chart(top_albums).mark_bar().encode(
            x=alt.X("play_count:Q", title="Plays"),
            y=alt.Y("album_name:N", sort="-x", title="Album"),
            tooltip=["album_name:N","play_count:Q"]
        ).properties(height=320, title="Top albums"), use_container_width=True)

        if "is_explicit" in curr.columns:
            exp_ct = curr["is_explicit"].fillna(False).value_counts().rename(index={True:"Explicit", False:"Clean"}).reset_index()
            exp_ct.columns = ["kind","count"]
            st.altair_chart(alt.Chart(exp_ct).mark_bar().encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("kind:N", sort="-x", title=""),
                tooltip=["kind:N","count:Q"]
            ).properties(height=140, title="Explicit vs Clean"), use_container_width=True)

        # Diversity curve
        curr_sorted = curr.sort_values("played_at").copy()
        if not curr_sorted.empty:
            curr_sorted["cum_unique_tracks"] = curr_sorted["track_id"].expanding().apply(lambda s: len(set(s)), raw=False)
            cum = curr_sorted[["played_at","cum_unique_tracks"]].copy()
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
            cB.metric("Avg duration (min)", round(sessions["session_duration_seconds"].mean() / 60, 1))
            cC.metric("Max tracks in a session", int(sessions["track_plays"].max()))
            st.dataframe(sessions, use_container_width=True, hide_index=True)

            st.altair_chart(
                alt.Chart(sessions.assign(duration_min=sessions["session_duration_seconds"]/60)).mark_bar().encode(
                    x=alt.X("duration_min:Q", bin=alt.Bin(maxbins=30), title="Duration (min)"),
                    y=alt.Y("count()", title="Sessions")
                ).properties(height=240, title="Session duration distribution"),
                use_container_width=True
            )

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

        span_days = (e - s).days + 1
        if span_days >= 28:
            by_day = curr.groupby("date").size().reset_index(name="plays")
            st.altair_chart(
                alt.Chart(by_day).mark_rect().encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.value(1),
                    color=alt.Color("plays:Q", title="Plays"),
                    tooltip=["date:T","plays:Q"]
                ).properties(height=60, title="Calendar strip (plays per day)"),
                use_container_width=True
            )

    # Audio & Genres
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
                st.altair_chart(
                    alt.Chart(merged.dropna(subset=["tempo"])).mark_bar().encode(
                        x=alt.X("tempo:Q", bin=alt.Bin(maxbins=40), title="Tempo (BPM)"),
                        y=alt.Y("count()", title="Tracks")
                    ).properties(height=220, title="Tempo distribution"),
                    use_container_width=True
                )
                st.altair_chart(
                    alt.Chart(merged.dropna(subset=["valence"])).mark_bar().encode(
                        x=alt.X("valence:Q", bin=alt.Bin(maxbins=30), title="Valence (positivity)"),
                        y=alt.Y("count()", title="Tracks")
                    ).properties(height=220, title="Valence (mood) distribution"),
                    use_container_width=True
                )
            else:
                st.info("No audio features returned (rate limited or no tracks).")

            with st.spinner("Fetching artist genres…"):
                gdf = sp_artists_genres(st.session_state["sp_access_token"], curr["artist_id"].dropna().unique().tolist())
            if not gdf.empty:
                top_genres = gdf["genre"].value_counts().reset_index()
                top_genres.columns = ["genre","count"]
                top_genres = top_genres.head(20)
                st.altair_chart(
                    alt.Chart(top_genres).mark_bar().encode(
                        x=alt.X("count:Q", title="Count"),
                        y=alt.Y("genre:N", sort="-x", title="Genre"),
                        tooltip=["genre:N","count:Q"]
                    ).properties(height=400, title="Top genres"),
                    use_container_width=True
                )
            else:
                st.info("No genres available for these artists.")


# ============================================================================
#                         ATHENA MODE (WAREHOUSE) HELPERS
# ============================================================================

def athena_mode_ui():
    st.subheader("🏗️ My Spotify Analytics")
    if not ATHENA_AVAILABLE:
        st.warning("`athena.py` not found or import failed — Athena mode disabled.")
        return

    # Controls
    default_end = dt.date.today()
    default_start = default_end - relativedelta(months=1)
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        start = st.date_input("Start date", value=default_start, key="start_date")
    with col_b:
        end = st.date_input("End date", value=default_end, key="end_date")

    # Schema helpers
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

    # Loaders (schema-aware)
    @st.cache_data(ttl=300)
    def load_kpis(start_date, end_date, dedupe_events: bool):
        s, e = str(start_date), str(end_date)
        cols = plays_cols()
        time_col = plays_time_expr(cols)
        distinct_kw = "DISTINCT " if dedupe_events else ""
        if time_col in ("played_at", "played_at_ts"):
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
        distinct_kw = "DISTINCT " if dedupe_events else ""
        if time_col in ("played_at", "played_at_ts"):
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
        distinct_kw = "DISTINCT " if dedupe_events else ""
        if time_col in ("played_at", "played_at_ts"):
            dedup_cte = f"""
            WITH dedup AS (
              SELECT {distinct_kw}{time_col}, track_id
              FROM {SCHEMA}.stg_spotify__plays
              WHERE DATE({time_col}) BETWEEN DATE '{s}' AND DATE '{e}'
            )
            """
        else:
            dedup_cte = f"""
            WITH dedup AS (
              SELECT {distinct_kw}dt, track_id
              FROM {SCHEMA}.stg_spotify__plays
              WHERE dt BETWEEN DATE '{s}' AND DATE '{e}'
            )
            """
        # Try join to dim_tracks first
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
        # Fallback
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
        # guard fact table
        q_exists = f"""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = '{SCHEMA}'
          AND table_name   = 'fct_listening_sessions'
        LIMIT 1
        """
        ex = read_sql(q_exists)
        if ex.empty:
            return pd.DataFrame(columns=["session_id","session_start","session_end","track_plays","session_duration_seconds"])
        sql = f"""
        SELECT session_id, session_start, session_end, track_plays, session_duration_seconds
        FROM {SCHEMA}.fct_listening_sessions
        WHERE DATE(session_start) BETWEEN DATE '{s}' AND DATE '{e}'
        ORDER BY session_start DESC
        LIMIT 1000
        """
        return read_sql(sql)

    # KPIs
    try:
        kpis = load_kpis(start, end, dedupe)
        c1, c2, c3, c4 = st.columns(4)
        if not kpis.empty:
            r = kpis.iloc[0].fillna(0)
            c1.metric("Total Plays", int(r.get("total_plays", 0) or 0))
            c2.metric("Unique Tracks", int(r.get("total_unique_tracks", 0) or 0))
            c3.metric("Avg Plays / Day", round(float(r.get("avg_plays_per_day", 0) or 0), 2))
            c4.metric("Active Days", int(r.get("active_days", 0) or 0))
            # Top track caption
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
        st.error("Failed to load KPIs from Athena.")
        st.exception(e)

    st.divider()

    # Trends
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

    # Top tracks
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

    # Sessions
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


# ============================================================================
#                              MODE TOGGLE & ROUTER
# ============================================================================

mode = st.radio("Mode", ["Live mode (user connects Spotify)", "Demo mode"], horizontal=True)

if mode == "Live mode (user connects Spotify)":
    live_mode_ui()
else:
    athena_mode_ui()

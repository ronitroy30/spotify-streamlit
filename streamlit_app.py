
import os
import time
import itertools
import urllib.parse
import datetime as dt

import pandas as pd
import requests
import streamlit as st
import altair as alt

# =========================
# App config
# =========================
st.set_page_config(page_title="Spotify Analytics — Live", layout="wide")

# Local timezone for display and day boundaries
TZ = "America/New_York"
def now_local():
    return pd.Timestamp.now(tz=TZ)

# =========================
# Spotify OAuth helpers
# =========================
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

# =========================
# Spotify data fetch
# =========================
def sp_fetch_recently_played(access_token: str, limit=50, pages=None, days_back: int = 30):
    """
    Walks the cursor 'before' to pull deeper history (up to days_back).
    Spotify caps the endpoint, but this goes far beyond a single page.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    base_url = "https://api.spotify.com/v1/me/player/recently-played"
    items, seen_ids = [], set()

    # Cutoff in UTC (endpoint uses UTC ms)
    cutoff = (now_local() - pd.Timedelta(days=days_back)).tz_convert("UTC")
    cutoff_ms = int(cutoff.timestamp() * 1000)

    before = None
    while True:
        params = {"limit": min(limit, 50)}
        if before is not None:
            params["before"] = before
        r = requests.get(base_url, headers=headers, params=params, timeout=30)
        if r.status_code in (401, 403):
            try:
                err = r.json()
            except Exception:
                err = {"error": {"status": r.status_code, "message": r.text}}
            raise RuntimeError(f"Spotify API {r.status_code}: {err}")
        r.raise_for_status()
        data = r.json()
        batch = data.get("items", [])
        if not batch:
            break

        # Stop if we’ve reached past cutoff
        oldest = batch[-1].get("played_at")
        if oldest:
            oldest_ms = int(pd.to_datetime(oldest, utc=True).timestamp() * 1000)
            if oldest_ms < cutoff_ms:
                # Keep only within window
                def keep(b):
                    ts = pd.to_datetime(b.get("played_at"), utc=True)
                    return int(ts.timestamp() * 1000) >= cutoff_ms
                batch = [b for b in batch if keep(b)]
                if not batch:
                    break

        # de-dupe by item id if present, else by (played_at, track.id)
        for it in batch:
            pid = it.get("id") or (it.get("played_at"), (it.get("track") or {}).get("id"))
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            items.append(it)

        # Move the cursor back
        cursor = data.get("cursors", {})
        before = cursor.get("before")
        if not before:
            nxt = data.get("next")
            if not nxt:
                break
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(nxt).query)
            before = qs.get("before", [None])[0]
            if not before:
                break

    return items

# =========================
# Normalization + Enrichment
# =========================
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
        # Parse as UTC then convert to local tz so dates/hours match your day
        df["played_at"] = pd.to_datetime(df["played_at"], utc=True).dt.tz_convert(TZ)
        # Clip any accidental “future” timestamps
        df = df[df["played_at"] <= now_local()]
        # Derive local day/hour fields after conversion
        df["date"] = df["played_at"].dt.date
        df["hour"] = df["played_at"].dt.hour
        df["weekday"] = df["played_at"].dt.day_name()
    return df

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
    # dedup while preserving order
    seen = set()
    de_duped = []
    for t in track_ids:
        if t and t not in seen:
            seen.add(t)
            de_duped.append(t)
    for batch in chunked(de_duped, 100):
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
    # dedup while preserving order
    seen = set()
    de_duped = []
    for a in artist_ids:
        if a and a not in seen:
            seen.add(a)
            de_duped.append(a)
    for batch in chunked(de_duped, 50):
        r = requests.get("https://api.spotify.com/v1/artists",
                         headers=headers, params={"ids": ",".join(batch)}, timeout=30)
        if r.status_code in (401,403):
            break
        r.raise_for_status()
        for a in r.json().get("artists", []):
            for g in (a.get("genres") or []):
                rows.append({"artist_id": a.get("id"), "genre": g})
    return pd.DataFrame(rows)

# =========================
# UI — Live mode only
# =========================
st.title("🎧 Spotify Analytics (Live)")

# Handle OAuth callback
qp = st.query_params
code = qp.get("code", [None])[0] if isinstance(qp.get("code"), list) else qp.get("code")

if "sp_access_token" not in st.session_state:
    st.session_state.update(sp_access_token=None, sp_refresh_token=None, sp_token_expiry=0, sp_scope="")

def need_refresh():
    return time.time() > st.session_state.get("sp_token_expiry", 0) - 60

# Exchange code → tokens
if code and not st.session_state["sp_access_token"]:
    try:
        tokens = sp_exchange_code_for_tokens(code)
        st.session_state["sp_access_token"] = tokens["access_token"]
        st.session_state["sp_refresh_token"] = tokens.get("refresh_token", st.session_state.get("sp_refresh_token"))
        st.session_state["sp_token_expiry"] = time.time() + int(tokens.get("expires_in", 3600))
        st.session_state["sp_scope"] = tokens.get("scope","")
        st.success("Spotify authorized!")
        st.caption(f"Granted scopes: {st.session_state['sp_scope']}")
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
            st.caption(f"Scopes: {st.session_state['sp_scope']}")
with c2:
    if st.session_state["sp_access_token"]:
        if st.button("Disconnect"):
            for k in ["sp_access_token","sp_refresh_token","sp_token_expiry","sp_scope"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()

if not st.session_state["sp_access_token"]:
    st.info("Connect Spotify to see your analytics.")
    st.stop()

# Refresh token if needed
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
        st.stop()

# How deep to fetch
days_back = st.slider("History depth (days)", 1, 90, 30,
                      help="How far back to fetch from Spotify’s 'Recently Played' endpoint.")

# Fetch & normalize
try:
    with st.spinner("Fetching your recent plays…"):
        items = sp_fetch_recently_played(st.session_state["sp_access_token"], days_back=days_back)
        df = normalize_recently_played(items)
except Exception as e:
    st.error("Could not fetch recently played.")
    st.exception(e)
    st.info("Ensure your account is a tester (Dev mode) and scope user-read-recently-played is granted.")
    st.stop()

if df.empty:
    st.info("No recent plays returned by Spotify. Play something and refresh.")
    st.stop()

# Single date range (local)
min_data_date = pd.to_datetime(df["date"]).min().date()
max_data_date = pd.to_datetime(df["date"]).max().date()
broad_min = (pd.to_datetime(min_data_date) - pd.Timedelta(days=60)).date()
broad_max = max(now_local().date(), max_data_date)

cA, cB = st.columns([1,1])
with cA:
    s = st.date_input("Start", value=min_data_date, min_value=broad_min, max_value=broad_max, key="live_s")
with cB:
    e = st.date_input("End", value=max_data_date, min_value=broad_min, max_value=broad_max, key="live_e")

# Filter to picked range
curr = df[(df["date"] >= s) & (df["date"] <= e)].copy()
if curr.empty:
    st.info("No plays in the selected range.")
    st.stop()

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

    # Diversity curve (no expanding.apply; robust to non-numeric)
    curr_sorted = curr.sort_values("played_at").copy()
    s_tracks = curr_sorted["track_id"].astype("string").fillna("_NULL_")
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
        st.altair_chart(
            alt.Chart(sessions.assign(duration_min=sessions["session_duration_seconds"]/60)).mark_bar().encode(
                x=alt.X("duration_min:Q", bin=alt.Bin(maxbins=30), title="Duration (min)"),
                y=alt.Y("count()", title="Sessions")
            ).properties(height=240, title="Session duration distribution"),
            use_container_width=True
        )

# Heatmaps
with tab4:
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    # Build full 24x7 grid
    full = (
        pd.MultiIndex.from_product([weekday_order, list(range(24))], names=["weekday","hour"])
        .to_frame(index=False)
    )
    # Actual counts from current selection
    heat_actual = curr.groupby(["weekday","hour"]).size().reset_index(name="plays")
    # Merge → fill missing combos with 0
    heat = full.merge(heat_actual, on=["weekday","hour"], how="left").fillna({"plays": 0})
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

# Audio & Genres (optional)
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

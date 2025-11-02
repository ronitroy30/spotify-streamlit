# Spotify Analytics — Streamlit Cloud

A shareable Streamlit app that reads **dbt** outputs from **AWS Athena** and visualizes KPIs, trends, top tracks, and listening sessions.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repo.
2. Go to https://streamlit.io/cloud → **Deploy an app** → select your repo, main file: `streamlit_app.py`.
3. In **App → Settings → Secrets**, paste:

```toml
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_DEFAULT_REGION = "us-east-1"
# If using temporary creds (STS/SSO):
# AWS_SESSION_TOKEN = "..."

ATHENA_WORKGROUP = "spotify_analytics_wg"
ATHENA_S3_STAGING = "s3://spotify-analytics-ronitroy-pipeline/athena-staging/"
ATHENA_SCHEMA = "spotify_analytics"


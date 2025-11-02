import os
import pandas as pd
from sqlalchemy.engine import create_engine

def _get(var, default=None):
    try:
        import streamlit as st
        if hasattr(st, "secrets") and var in st.secrets:
            return st.secrets[var]
    except Exception:
        pass
    return os.getenv(var, default)

def get_engine():
    region = _get("AWS_DEFAULT_REGION", "us-east-1")
    workgroup = _get("ATHENA_WORKGROUP", "spotify_analytics_wg")
    s3_staging = _get("ATHENA_S3_STAGING")
    if not s3_staging:
        raise RuntimeError("ATHENA_S3_STAGING not set")

    conn = (
        f"awsathena+rest://@athena.{region}.amazonaws.com/"
        f"awsdatacatalog?work_group={workgroup}&s3_staging_dir={s3_staging}"
    )
    return create_engine(conn)

def read_sql(sql, params=None) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as con:
        return pd.read_sql(sql, con=con, params=params or {})

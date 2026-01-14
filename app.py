
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dateutil import parser
from supabase import create_client

# ---- Supabase credentials (set these in Colab BEFORE importing app.py) ----
# Recommended: keep these in environment variables (safer than hardcoding)
import os
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
analyzer = SentimentIntensityAnalyzer()

def load_comments_supabase(limit: int = 2000) -> pd.DataFrame:
    resp = (
        supabase
        .table("comments")
        .select("comment_id, video_id, stance, model, text_original, like_count, comment_published_at")
        .order("comment_published_at", desc=True)
        .limit(limit)
        .execute()
    )
    df = pd.DataFrame(resp.data or [])
    if df.empty:
        return df
    df["comment_published_at"] = df["comment_published_at"].apply(lambda x: parser.isoparse(x) if pd.notnull(x) else None)
    df = df.dropna(subset=["text_original", "comment_published_at"])
    df["text_original"] = df["text_original"].astype(str).str.strip()
    df = df[df["text_original"].str.len() > 0]
    return df

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_compound"] = df["text_original"].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
    df["sentiment_label"] = pd.cut(
        df["sentiment_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"]
    )
    return df

def plot_sentiment_over_time(df: pd.DataFrame, freq: str = "D"):
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("No data available")
        return fig

    dfx = df.copy()
    dfx["comment_published_at"] = pd.to_datetime(dfx["comment_published_at"], utc=True, errors="coerce")
    dfx = dfx.dropna(subset=["comment_published_at", "sentiment_compound"])
    dfx["time_bucket"] = dfx["comment_published_at"].dt.to_period(freq).dt.to_timestamp()

    ts = (
        dfx.groupby("time_bucket")["sentiment_compound"]
        .mean()
        .reset_index()
        .sort_values("time_bucket")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts["time_bucket"], ts["sentiment_compound"])
    ax.set_title("Sentiment over Time (avg VADER compound)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Avg Sentiment")
    ax.grid(True)
    fig.autofmt_xdate()
    return fig

def drilldown_top2(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    dfx = df.dropna(subset=["text_original", "sentiment_compound"]).copy()
    most_pos = dfx.sort_values("sentiment_compound", ascending=False).head(2)
    most_neg = dfx.sort_values("sentiment_compound", ascending=True).head(2)

    cols = ["comment_id","video_id","stance","model","sentiment_compound","text_original"]
    return most_pos[cols], most_neg[cols]

def run_dashboard(stance_filter: str, freq: str, limit: int):
    df = load_comments_supabase(limit=limit)
    if df.empty:
        fig = plot_sentiment_over_time(df, freq="D")
        return fig, 0, 0, 0

    df = add_sentiment(df)

    if stance_filter != "all":
        df = df[df["stance"] == stance_filter]

    fig = plot_sentiment_over_time(df, freq=freq)

    total = len(df)
    pos = int((df["sentiment_label"] == "positive").sum())
    neg = int((df["sentiment_label"] == "negative").sum())

    return fig, total, pos, neg

def run_drilldown(stance_filter: str, limit: int):
    df = load_comments_supabase(limit=limit)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = add_sentiment(df)
    if stance_filter != "all":
        df = df[df["stance"] == stance_filter]

    pos2, neg2 = drilldown_top2(df)
    return pos2, neg2

def build_app():
    with gr.Blocks(title="Market Pulse Engine, Tesla") as demo:
        gr.Markdown("# 📈 Market Pulse Engine, Tesla")
        gr.Markdown("Reads from Supabase → sentiment → sentiment-over-time → drill-down.")

        with gr.Tab("Dashboard"):
            stance = gr.Dropdown(["all", "fanboy", "critical"], value="all", label="Filter by stance")
            freq = gr.Dropdown(["D", "W", "M"], value="D", label="Time bucket (D=day, W=week, M=month)")
            limit = gr.Slider(200, 5000, value=2000, step=100, label="Recent comments to analyze")
            run_btn = gr.Button("Run Dashboard")

            plot = gr.Plot(label="Sentiment-over-Time")
            total_kpi = gr.Number(label="Total comments analyzed")
            pos_kpi = gr.Number(label="Positive count")
            neg_kpi = gr.Number(label="Negative count")

            run_btn.click(fn=run_dashboard, inputs=[stance, freq, limit], outputs=[plot, total_kpi, pos_kpi, neg_kpi])

        with gr.Tab("Drill-Down"):
            stance2 = gr.Dropdown(["all", "fanboy", "critical"], value="all", label="Filter by stance")
            limit2 = gr.Slider(200, 5000, value=2000, step=100, label="Recent comments to analyze")
            dd_btn = gr.Button("Find Top 2 Positive/Negative")

            pos_table = gr.Dataframe(label="Top 2 Most Positive Comments", wrap=True)
            neg_table = gr.Dataframe(label="Top 2 Most Negative Comments", wrap=True)

            dd_btn.click(fn=run_drilldown, inputs=[stance2, limit2], outputs=[pos_table, neg_table])

    return demo

if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise ValueError("Missing SUPABASE_URL / SUPABASE_ANON_KEY env vars.")
    app = build_app()
    app.launch(share=True)

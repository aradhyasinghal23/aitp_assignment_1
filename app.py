import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dateutil import parser

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
    df["comment_published_at"] = df["comment_published_at"].apply(
        lambda x: parser.isoparse(x) if pd.notnull(x) else None
    )
    df = df.dropna(subset=["text_original", "comment_published_at"])
    df["text_original"] = df["text_original"].astype(str).str.strip()
    df = df[df["text_original"].str.len() > 0]
    return df

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_compound"] = df["text_original"].apply(
        lambda t: analyzer.polarity_scores(str(t))["compound"]
    )
    df["sentiment_label"] = pd.cut(
        df["sentiment_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"]
    )
    return df

def plot_sentiment_over_time(df: pd.DataFrame, freq: str = "D"):
    if df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_title("No data available")
        ax.axis("off")
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

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ts["time_bucket"], ts["sentiment_compound"])
    ax.set_title("Sentiment over Time (avg VADER compound)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Avg Sentiment")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig

def drilldown_top2(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    dfx = df.dropna(subset=["text_original", "sentiment_compound"]).copy()
    most_pos = dfx.sort_values("sentiment_compound", ascending=False).head(2)
    most_neg = dfx.sort_values("sentiment_compound", ascending=True).head(2)

    cols = ["comment_id", "video_id", "stance", "model", "sentiment_compound", "text_original"]
    return most_pos[cols], most_neg[cols]

def run_dashboard(stance_filter: str, freq: str, limit: int):
    df = load_comments_supabase(limit=limit)
    if df.empty:
        fig = plot_sentiment_over_time(df, freq="D")
        return fig, 0, 0, 0, 0

    df = add_sentiment(df)

    if stance_filter != "all":
        df = df[df["stance"] == stance_filter]

    fig = plot_sentiment_over_time(df, freq=freq)

    total = len(df)
    pos = int((df["sentiment_label"] == "positive").sum())
    neu = int((df["sentiment_label"] == "neutral").sum())
    neg = int((df["sentiment_label"] == "negative").sum())

    return fig, total, pos, neu, neg

def run_drilldown(stance_filter: str, limit: int):
    df = load_comments_supabase(limit=limit)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = add_sentiment(df)
    if stance_filter != "all":
        df = df[df["stance"] == stance_filter]

    pos2, neg2 = drilldown_top2(df)
    return pos2, neg2


# --- Light UI polish (not fancy, just cleaner) ---
css = """
:root { --radius: 12px; }
#titlebar h1 { margin-bottom: 0.25rem; }
.kpi .gr-number { text-align: center; }
"""

with gr.Blocks(
    title="Market Pulse Engine — Tesla",
    theme=gr.themes.Soft(),
    css=css
) as demo:

    # Header row
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown(
                """
                <div id="titlebar">
                  <h1>📈 Market Pulse Engine — Tesla</h1>
                  <p style="opacity:0.8; margin-top:0;">
                    Supabase → VADER sentiment → trend view → drill-down into extremes.
                  </p>
                </div>
                """
            )
        with gr.Column(scale=4, min_width=260):
            gr.Markdown(
                """
                **Quick guide**
                - Use *Dashboard* for the sentiment trend  
                - Use *Drill-Down* to inspect extreme comments  
                """
            )

    with gr.Tabs():

        # -------------------------
        # DASHBOARD TAB
        # -------------------------
        with gr.TabItem("Dashboard", id="dash"):
            with gr.Row():

                # Controls panel
                with gr.Column(scale=3, min_width=260):
                    with gr.Group():
                        gr.Markdown("### Controls")
                        stance = gr.Dropdown(
                            ["all", "fanboy", "critical"],
                            value="all",
                            label="Stance filter",
                            info="All = mixes both fanboy + critical videos."
                        )
                        freq = gr.Radio(
                            ["D", "W", "M"],
                            value="D",
                            label="Time bucket",
                            info="D=day, W=week, M=month"
                        )
                        limit = gr.Slider(
                            200, 5000,
                            value=2000,
                            step=100,
                            label="Recent comments to analyze",
                            info="Higher = smoother trend, slower load."
                        )
                        run_btn = gr.Button("Run Dashboard", variant="primary")

                # Plot + KPIs panel
                with gr.Column(scale=7):
                    with gr.Group():
                        gr.Markdown("### Trend")
                        plot = gr.Plot(label="Avg sentiment over time")

                    with gr.Row():
                        total_kpi = gr.Number(label="Total", elem_classes=["kpi"])
                        pos_kpi = gr.Number(label="Positive", elem_classes=["kpi"])
                        neu_kpi = gr.Number(label="Neutral", elem_classes=["kpi"])
                        neg_kpi = gr.Number(label="Negative", elem_classes=["kpi"])

            run_btn.click(
                fn=run_dashboard,
                inputs=[stance, freq, limit],
                outputs=[plot, total_kpi, pos_kpi, neu_kpi, neg_kpi]
            )

        # -------------------------
        # DRILL-DOWN TAB
        # -------------------------
        with gr.TabItem("Drill-Down", id="dd"):
            with gr.Row():

                # Controls panel
                with gr.Column(scale=3, min_width=260):
                    with gr.Group():
                        gr.Markdown("### Controls")
                        stance2 = gr.Dropdown(
                            ["all", "fanboy", "critical"],
                            value="all",
                            label="Stance filter"
                        )
                        limit2 = gr.Slider(
                            200, 5000,
                            value=2000,
                            step=100,
                            label="Recent comments to analyze"
                        )
                        dd_btn = gr.Button("Find Top 2 Extremes", variant="primary")
                        gr.Markdown(
                            "<span style='opacity:0.75'>Ranks by VADER compound score.</span>"
                        )

                # Tables panel
                with gr.Column(scale=7):
                    with gr.Row():
                        pos_table = gr.Dataframe(
                            label="Top 2 Most Positive",
                            wrap=True,
                            interactive=False,
                            row_count=2
                        )
                        neg_table = gr.Dataframe(
                            label="Top 2 Most Negative",
                            wrap=True,
                            interactive=False,
                            row_count=2
                        )

            dd_btn.click(
                fn=run_drilldown,
                inputs=[stance2, limit2],
                outputs=[pos_table, neg_table]
            )

demo.launch(share=True)

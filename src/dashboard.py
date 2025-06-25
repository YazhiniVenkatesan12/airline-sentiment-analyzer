# src/dashboard.py
import pandas as pd
import streamlit as st
import altair as alt
from datetime import date

# ─────────────────────────  CONFIG  ─────────────────────────
st.set_page_config(
    page_title="Airline Tweets – Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject minimal CSS
st.markdown(
    """
    <style>
    /* Hide Streamlit default footer & menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Global font */
    html, body, [class*="css"]  {font-family: 'Segoe UI', sans-serif;}
    /* KPI card tweaks */
    .metric {border-radius: 8px; padding: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Banner
st.markdown(
    """
    <h1 style='text-align:center; color:#0E6BA8; margin-bottom:0'>
        ✈️ Airline Tweet Analytics Dashboard
    </h1>
    <p style='text-align:center; color:#555; margin-top:4px'>
        Real-time sentiment & sarcasm insights
    </p>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────  LOAD DATA  ─────────────────────────
CSV = os.path.join(os.path.dirname(__file__),"C:/Users/hp/Documents/PROJECT/sentiment-tweets/data/cleaned_tweets.csv")
df = pd.read_csv(CSV)

# Parse datetime safely
df["tweet_created"] = pd.to_datetime(
    df["tweet_created"], errors="coerce", utc=True
).dt.tz_convert(None)
df = df.dropna(subset=["tweet_created", "airline_sentiment", "text"])

# Sarcasm keywords
SARCASTIC = [
    "grateful",
    "blessed",
    "thanks a lot",
    "shoutout",
    "🙃",
    "so happy",
    "awesome work",
]
df["sarcasm_flag"] = df["text"].str.lower().apply(
    lambda x: any(k in x for k in SARCASTIC)
)

# ─────────────────────────  SIDEBAR FILTERS  ─────────────────────────
st.sidebar.header("📂 Filters")

# Date range
min_d, max_d = df["tweet_created"].min().date(), df["tweet_created"].max().date()
start, end = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)
if start > end:
    st.sidebar.error("Start date must be before end date")
    st.stop()

# Airline multiselect
airlines_all = sorted(df["airline"].dropna().unique())
choose_airlines = st.sidebar.multiselect(
    "Airlines",
    airlines_all,
    default=airlines_all,
)

# Apply filters
mask = (df["tweet_created"].dt.date.between(start, end)) & (
    df["airline"].isin(choose_airlines)
)
data = df[mask]

# ─────────────────────────  KPI METRICS  ─────────────────────────
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Tweets", f"{len(data):,}")
kpi2.metric("Distinct Airlines", f"{data['airline'].nunique()}")
kpi3.metric(
    "Possible Sarcasm",
    f"{data['sarcasm_flag'].sum():,}",
    delta=f"{data['sarcasm_flag'].mean()*100:.1f}%",
)

st.markdown("---")

# ─────────────────────────  TAB LAYOUT  ─────────────────────────
tabs = st.tabs(["📊 Overview", "📈 Activity", "✈️ Airline Split"])

# ─── TAB 1: OVERVIEW ───
with tabs[0]:
    st.subheader("Sentiment Breakdown")
    overview_chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("count()", title="Tweet Count"),
            y=alt.Y("airline_sentiment:N", sort="-x", title=" "),
            color=alt.Color(
                "airline_sentiment",
                scale=alt.Scale(
                    domain=["negative", "neutral", "positive"],
                    range=["#E45756", "#F4D35E", "#4CAF50"],
                ),
                legend=None,
            ),
            tooltip=["count()"],
        )
        .properties(height=200)
    )
    st.altair_chart(overview_chart, use_container_width=True)

    st.subheader("Recent Sarcasm-like Tweets")
    st.dataframe(
        data.loc[data["sarcasm_flag"], ["tweet_created", "airline", "text"]]
        .sort_values("tweet_created", ascending=False)
        .head(10)
        .rename(
            columns={
                "tweet_created": "Date",
                "airline": "Airline",
                "text": "Tweet",
            }
        ),
        hide_index=True,
    )

# ─── TAB 2: ACTIVITY ───
with tabs[1]:
    st.subheader("Daily Tweet Volume by Sentiment")
    daily = (
        data.groupby([data["tweet_created"].dt.date, "airline_sentiment"])
        .size()
        .reset_index(name="count")
    )

    line = (
        alt.Chart(daily)
        .mark_line(interpolate="monotone", strokeWidth=2)
        .encode(
            x=alt.X("tweet_created:T", title="Date"),
            y=alt.Y("count:Q", title="Tweets / day"),
            color="airline_sentiment",
            tooltip=["tweet_created:T", "count", "airline_sentiment"],
        )
        .interactive()
        .properties(height=300)
    )
    st.altair_chart(line, use_container_width=True)

# ─── TAB 3: AIRLINE SPLIT ───
with tabs[2]:
    st.subheader("Sentiment by Airline (Stacked %)")

    airline_sent = (
        data.groupby(["airline_sentiment", "airline"])
        .size()
        .reset_index(name="count")
    )
    total_airline = airline_sent.groupby("airline")["count"].transform("sum")
    airline_sent["percent"] = airline_sent["count"] / total_airline

    bar = (
        alt.Chart(airline_sent)
        .mark_bar()
        .encode(
            x=alt.X("airline:N", title="Airline", sort="-y"),
            y=alt.Y("percent:Q", axis=alt.Axis(format="%"), title="Share"),
            color="airline_sentiment",
            tooltip=["airline_sentiment", alt.Tooltip("percent:Q", format=".1%")],
        )
        .properties(height=400)
    )
    st.altair_chart(bar, use_container_width=True)


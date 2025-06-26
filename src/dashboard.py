# src/dashboard.py
import os
import pandas as pd
import streamlit as st
import altair as alt
from datetime import date

# ─────────── CONFIG ───────────
st.set_page_config(page_title="Airline Tweets – Analytics", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    html, body, [class*="css"] {font-family: 'Segoe UI', sans-serif;}
    .metric {border-radius: 8px; padding: 8px;}
    .dark-toggle {
        position: absolute; top: 1rem; right: 2rem; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────── Dark Mode Toggle ────────────────
with st.container():
    col1, col2 = st.columns([0.05, 0.95])
    with col1:
        dark_icon = "🌙" if not st.session_state.get("dark", False) else "☀️"
        st.markdown(f"<h4 style='margin:0'>{dark_icon}</h4>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h5 style='margin:0.3em 0 0 0;'>Toggle Theme</h5>", unsafe_allow_html=True)
        st.session_state.dark = st.toggle(" ", label_visibility="collapsed", value=st.session_state.get("dark", False))

# Apply dark mode styling
if st.session_state.dark:
    st.markdown("""
        <style>
            html { filter: invert(1) hue-rotate(180deg); }
            img, video { filter: invert(1) hue-rotate(180deg); }
        </style>
    """, unsafe_allow_html=True)


# ─────────── TITLE ───────────
st.markdown("""
    <h1 style='text-align:center; color:#0E6BA8; margin-bottom:0'>
        ✈️ Airline Tweet Analytics Dashboard
    </h1>
    <p style='text-align:center; color:#555; margin-top:4px'>
        Real-time sentiment & sarcasm insights
    </p>
""", unsafe_allow_html=True)

# ─────────── LOAD DATA ───────────
CSV = os.path.join("data", "cleaned_tweets.csv")
df = pd.read_csv(CSV)

df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors="coerce", utc=True).dt.tz_convert(None)
df = df.dropna(subset=["tweet_created", "airline_sentiment", "text"])

SARCASTIC = ["grateful", "blessed", "thanks a lot", "shoutout", "🙃", "so happy", "awesome work"]
df["sarcasm_flag"] = df["text"].str.lower().apply(lambda x: any(k in x for k in SARCASTIC))

# ─────────── FILTERS ───────────
st.markdown("### 🔍 Filter Results")
min_d, max_d = df["tweet_created"].min().date(), df["tweet_created"].max().date()
col1, col2, col3 = st.columns([3, 3, 4])

with col1:
    start = st.date_input("Start Date", value=min_d, min_value=min_d, max_value=max_d)
with col2:
    end = st.date_input("End Date", value=max_d, min_value=min_d, max_value=max_d)
with col3:
    airlines_all = sorted(df["airline"].dropna().unique())
    choose_airlines = st.multiselect("Select Airlines", airlines_all, default=airlines_all)

if start > end:
    st.error("❌ Start date must be before end date.")
    st.stop()

mask = (df["tweet_created"].dt.date.between(start, end)) & (df["airline"].isin(choose_airlines))
data = df[mask]

# ─────────── KPI METRICS ───────────
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Tweets", f"{len(data):,}")
kpi2.metric("Distinct Airlines", f"{data['airline'].nunique()}")
kpi3.metric("Possible Sarcasm", f"{data['sarcasm_flag'].sum():,}", delta=f"{data['sarcasm_flag'].mean()*100:.1f}%")

st.markdown("---")

# ─────────── TABS ───────────
tabs = st.tabs(["📊 Overview", "📈 Activity", "✈️ Airline Split"])

# ─── TAB 1: OVERVIEW ───
with tabs[0]:
    st.subheader("Sentiment Breakdown")
    overview_chart = alt.Chart(data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("count()", title="Tweet Count"),
        y=alt.Y("airline_sentiment:N", sort="-x", title=" "),
        color=alt.Color(
            "airline_sentiment",
            scale=alt.Scale(domain=["negative", "neutral", "positive"],
                            range=["#E45756", "#F4D35E", "#4CAF50"]),
            legend=None,
        ),
        tooltip=["count()"]
    ).properties(height=200)
    st.altair_chart(overview_chart, use_container_width=True)

    st.subheader("Recent Sarcasm-like Tweets")
    st.dataframe(
        data.loc[data["sarcasm_flag"], ["tweet_created", "airline", "text"]]
        .sort_values("tweet_created", ascending=False)
        .head(10)
        .rename(columns={"tweet_created": "Date", "airline": "Airline", "text": "Tweet"}),
        hide_index=True,
    )

# ─── TAB 2: ACTIVITY ───
with tabs[1]:
    st.subheader("Daily Tweet Volume by Sentiment")
    daily = data.groupby([data["tweet_created"].dt.date, "airline_sentiment"]).size().reset_index(name="count")

    line = alt.Chart(daily).mark_line(interpolate="monotone", strokeWidth=2).encode(
        x=alt.X("tweet_created:T", title="Date"),
        y=alt.Y("count:Q", title="Tweets / day"),
        color="airline_sentiment",
        tooltip=["tweet_created:T", "count", "airline_sentiment"],
    ).interactive().properties(height=300)

    st.altair_chart(line, use_container_width=True)

# ─── TAB 3: AIRLINE SPLIT ───
with tabs[2]:
    st.subheader("Sentiment by Airline (Stacked %)")

    airline_sent = data.groupby(["airline_sentiment", "airline"]).size().reset_index(name="count")
    total_airline = airline_sent.groupby("airline")["count"].transform("sum")
    airline_sent["percent"] = airline_sent["count"] / total_airline

    bar = alt.Chart(airline_sent).mark_bar().encode(
        x=alt.X("airline:N", title="Airline", sort="-y"),
        y=alt.Y("percent:Q", axis=alt.Axis(format="%"), title="Share"),
        color="airline_sentiment",
        tooltip=["airline_sentiment", alt.Tooltip("percent:Q", format=".1%")],
    ).properties(height=400)

    st.altair_chart(bar, use_container_width=True)
    # ─────────── FOOTER LINKS ───────────
st.markdown(
    """
    <hr style='margin-top:2rem;'>
    <div style='text-align:center; font-size:0.9rem; color:#666;'>
        🌐 <a href="https://airline-sentiment-analyzer.streamlit.app" target="_blank">View Live App</a> •
        💻 <a href="https://github.com/YazhiniVenkatesan12/airline-sentiment-analyzer" target="_blank">Source Code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)


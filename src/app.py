# src/app.py
import os, re, datetime as dt
import streamlit as st
import nltk, torch
from nltk.corpus import stopwords
from streamlit.components.v1 import html
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ──────────────────────────────────────────────
# 1. NLTK resources
nltk.download("stopwords")
STOP = set(stopwords.words("english"))
CLEAN_RE = re.compile(r"http\S+|@\S+|#\S+|[^A-Za-z\s]")

def clean(text: str) -> str:
    return " ".join(
        w for w in CLEAN_RE.sub("", text.lower()).split() if w not in STOP
    )

# ──────────────────────────────────────────────
# 2. Sentiment model (RoBERTa)
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)

model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL,
    low_cpu_mem_usage=False,   # avoid meta tensors
    torch_dtype="auto",        # loads fp16 if available
    trust_remote_code=True     # safe for HF community models
).to("cpu")                    # explicit device
model.eval()
LABELS    = ["negative", "neutral", "positive"]

# ──────────────────────────────────────────────
# 3. Sarcasm model (fast RoBERTa)
sarcasm_model = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-irony",
    tokenizer="cardiffnlp/twitter-roberta-base-irony",
    device=0 if torch.cuda.is_available() else -1,
    use_fast=True,
    trust_remote_code=True,
    revision="main",
)

# ──────────────────────────────────────────────
# 4. Page + global CSS
st.set_page_config(page_title="Airline Tweet Analyzer", page_icon="✈️", layout="centered",initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        /* Hide default Streamlit clutter */
        #MainMenu, footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Global font & palette */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: #f4f8fb;
        }

        /* Brand header */
        .title-wrapper {text-align:center; margin-top:-10px;}
        .title-wrapper h1 {color:#0E6BA8; margin-bottom:0;}
        .title-wrapper p  {color:#555; margin-top:4px;}

        /* Rounded text area */
        textarea {
            border:1px solid #d0d7de !important; border-radius:10px !important;
            padding:1rem !important; font-size:1rem !important;
        }

        /* Buttons */
        .stButton>button {
            background:#0E6BA8; color:#fff; border:none; border-radius:8px;
            padding:0.6em 1.4em; transition:all .3s;
            box-shadow:0 3px 6px rgba(0,0,0,.1);
        }
        .stButton>button:hover {
            background:#5DB0FF; transform:translateY(-2px);
            box-shadow:0 6px 18px rgba(0,98,180,.25);
        }

        /* Result card */
        .result-card {
            background:#fff; border-radius:12px; padding:1.5rem;
            box-shadow:0 4px 14px rgba(0,0,0,.06);
            text-align:center;
        }
        .result-card h2 {font-size:2.6rem; margin:0.2em 0;}
        .emoji {font-size:2.8rem;}

        /* Progress bar tweaks */
        .stProgress>div>div>div {
            border-radius:10px; background:#0E6BA8;
        }

        /* Alert */
        .stAlert {border-radius:10px;}

        /* Selectbox label */
        .stSelectbox label {font-weight:600; color:#0E6BA8;}

    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────── 5. Brand banner ────────────
st.markdown(
    """
    <div class="title-wrapper">
        <h1>✈️ Airline Tweet Sentiment Analyzer</h1>
        <p>Powered by RoBERTa &amp; Sarcasm Detection</p>
    </div>
    """,
    unsafe_allow_html=True,
)
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

# ──────────────────────────────────────────────
# 6. Sample tweets + input
samples = {
    "Choose a sample…": "",
    "😊 Positive":  "Amazing flight today – staff were super friendly and helpful!",
    "😐 Neutral":   "My flight departs from gate B12 at 5:20 PM.",
    "😡 Negative":  "Flight delayed for 5 hours with zero explanation. Unacceptable!",
    "😏 Sarcastic": "Thanks @AirlineName for losing my luggage again. You're truly unmatched 🙃."
}
def reset_form():
    st.session_state.input_text  = ""
    st.session_state.sample_select = list(samples.keys())[0]

sel_key = st.selectbox("Quick test tweets (optional):", list(samples.keys()), key="sample_select")
default_text = samples[sel_key]

MAX_LEN = 280
tweet = st.text_area(
    "Type or paste a tweet:", value=default_text, height=140, max_chars=MAX_LEN, key="input_text"
)
st.write(f"✏️ {len(tweet)}/{MAX_LEN} characters")


# ──────────────────────────────────────────────
# 7. Buttons
col1, col2 = st.columns(2)
predict = col1.button("Predict Sentiment")
clear   = col2.button("Reset", on_click=reset_form)

# ──────────────────────────────────────────────
# 8. Prediction & UI output
if predict:
    txt = tweet.strip()
    if not txt:
        st.warning("Please enter a tweet 😊")
        st.stop()

    # Sentiment prediction
    with torch.no_grad():
        tokens = tokenizer(clean(txt), return_tensors="pt", truncation=True, padding=True)
        probs  = torch.softmax(model(**tokens).logits, dim=1).squeeze().tolist()

    scores     = torch.tensor(probs)        # convert list → Tensor
    label_idx  = int(torch.argmax(scores))  # now argmax works
    label     = LABELS[label_idx]
    conf      = probs[label_idx] * 100
    emoji     = {"positive":"😊","neutral":"😐","negative":"😡"}[label]

    # Result card
    emoji_html = f"<span class='emoji' role='img' aria-label='{label}'>{emoji}</span>"

    gauge_html = f"""
    <div style='display:flex; justify-content:center;'>
    <svg viewBox="0 0 36 36" width="110">
        <path d="M18 2.084a15.916 15.916 0 1 1 0 31.832"
            fill="none" stroke="#e6e6e6" stroke-width="3"/>
        <path d="M18 2.084a15.916 15.916 0 0 1 0 {(conf/100)*31.832}"
            fill="none" stroke="#0E6BA8" stroke-width="3" stroke-linecap="round"/>
        <text x="18" y="20.35" text-anchor="middle"
            fill="#0E6BA8" font-size="9" font-weight="600">{conf:.0f}%</text>
    </svg>
    </div>
    """

    result_html = f"""
    <div class="result-card">
        {emoji_html}
        <h2>{label.upper()}</h2>
        {gauge_html}
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)


    # Confidence breakdown
    with st.expander("Show confidence scores"):
        for lbl, p in zip(LABELS, probs):
            st.write(f"- **{lbl.capitalize():8}** : {p*100:.2f}%")

    # Sarcasm detection
    try:
        model_sarc = sarcasm_model(txt)[0]
        is_model_sarc = model_sarc['label'].lower() in ['irony','sarcasm'] and model_sarc['score'] > 0.5
        rule_clues = ["grateful","blessed","thanks a lot","great job","awesome","🙃","yay","love that","shoutout","exactly what I needed","so helpful"]
        is_rule_sarc = any(c in txt.lower() for c in rule_clues)

        if is_model_sarc or is_rule_sarc:
            st.warning("⚠️ This tweet may be sarcastic or ironic. Sentiment prediction might not reflect the true intent.")
    except Exception as e:
        st.error("Sarcasm detector error: " + str(e))

    # Log
    os.makedirs("../logs", exist_ok=True)
    with open("../logs/predictions.csv", "a", encoding="utf8") as f:
        ts = dt.datetime.utcnow().isoformat(timespec="seconds")
        f.write(f'"{ts}","{txt.replace(chr(34),chr(39))}","{label}",{conf:.2f}\n')

# ──────────────────────────────────────────────
# 9. Footer links
st.markdown(
    """
    <hr style='margin-top:2rem;'>
    <div style='text-align:center; font-size:0.9rem; color:#666;'>
        📊 View the <a href="https://airline-sentiment-analyzer-dashboard.streamlit.app" target="_blank"><b>Dashboard</b></a> for analytics. •
        GitHub repo → <a href="https://github.com/YazhiniVenkatesan12/airline-sentiment-analyzer" target="_blank">Source</a>
    </div>
    """,
    unsafe_allow_html=True
)


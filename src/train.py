# src/train.py

import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

# ─────────────────────────────
# Setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ─────────────────────────────
# Load Data
df = pd.read_csv('C:/Users/hp/Documents/PROJECT/sentiment-tweets/data/cleaned_tweets.csv')

# ─────────────────────────────
# Text Cleaning
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z\s]", "", text.lower())
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].astype(str).apply(clean_text)

# ─────────────────────────────
# Features & Labels
X = df['clean_text']
y = df['airline_sentiment']

# ─────────────────────────────
# TF-IDF + Model Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=7000, ngram_range=(1, 3))),
    ('clf', LogisticRegression(max_iter=300, C=0.5, solver='liblinear'))
])

# ─────────────────────────────
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ─────────────────────────────
# Train the Model
pipeline.fit(X_train, y_train)

# ─────────────────────────────
# Evaluate
y_pred = pipeline.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# ─────────────────────────────
# Save the Pipeline
os.makedirs('C:/Users/hp/Documents/PROJECT/sentiment-tweets/models', exist_ok=True)
joblib.dump(pipeline, 'C:/Users/hp/Documents/PROJECT/sentiment-tweets/models/sentiment_pipeline.pkl')
print("📦 Model saved to C:/Users/hp/Documents/PROJECT/sentiment-tweets/models/sentiment_pipeline.pkl")

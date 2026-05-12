import os
import re
import json
import random
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "journal_entries.csv"

CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "hopeless",
    "worthless",
    "die",
    "self harm",
    "depressed",
]

STRESS_KEYWORDS = [
    "stress",
    "anxiety",
    "worried",
    "panic",
    "fear",
    "tired",
    "burnout",
    "pressure",
]

POSITIVE_WORDS = [
    "happy",
    "great",
    "amazing",
    "good",
    "peaceful",
    "calm",
    "excited",
    "joy",
]

NEGATIVE_WORDS = [
    "sad",
    "angry",
    "depressed",
    "hopeless",
    "cry",
    "pain",
    "stress",
    "anxiety",
]


# ============================================================
# CREATE DATA FILE IF NOT EXISTS
# ============================================================

def initialize_file():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=[
            "date",
            "journal_text",
            "sentiment",
            "mood",
            "stress_score",
            "crisis_alert"
        ])
        df.to_csv(DATA_FILE, index=False)


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        mood = "Positive"
    elif polarity < -0.2:
        mood = "Negative"
    else:
        mood = "Neutral"

    return polarity, mood


# ============================================================
# STRESS PREDICTION
# ============================================================

def predict_stress(text):
    text_lower = text.lower()

    stress_count = 0

    for word in STRESS_KEYWORDS:
        if word in text_lower:
            stress_count += 1

    score = min(stress_count * 20, 100)

    if score >= 60:
        level = "High Stress"
    elif score >= 30:
        level = "Moderate Stress"
    else:
        level = "Low Stress"

    return score, level


# ============================================================
# CRISIS DETECTION
# ============================================================

def detect_crisis(text):
    text_lower = text.lower()

    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return True

    return False


# ============================================================
# SAVE ENTRY
# ============================================================

def save_entry(text):
    sentiment, mood = analyze_sentiment(text)
    stress_score, stress_level = predict_stress(text)
    crisis = detect_crisis(text)

    new_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "journal_text": text,
        "sentiment": sentiment,
        "mood": mood,
        "stress_score": stress_score,
        "crisis_alert": crisis
    }

    df = pd.read_csv(DATA_FILE)

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    df.to_csv(DATA_FILE, index=False)

    print("\n======================================")
    print("Journal Entry Saved Successfully")
    print("======================================")

    print(f"Mood Detected     : {mood}")
    print(f"Sentiment Score   : {round(sentiment, 2)}")
    print(f"Stress Score      : {stress_score}/100")
    print(f"Stress Level      : {stress_level}")

    if crisis:
        print("\n⚠️ CRISIS ALERT DETECTED ⚠️")
        print("Please consider contacting a trusted person or mental health professional.\n")

    print("======================================\n")


# ============================================================
# SHOW ANALYTICS
# ============================================================

def show_analytics():
    df = pd.read_csv(DATA_FILE)

    if df.empty:
        print("No journal data available.")
        return

    print("\n========== JOURNAL ANALYTICS ==========")

    print("\nTotal Entries:", len(df))

    mood_counts = df["mood"].value_counts()

    print("\nMood Distribution:")
    print(mood_counts)

    avg_stress = df["stress_score"].mean()

    print(f"\nAverage Stress Score: {round(avg_stress, 2)}")

    crisis_count = df["crisis_alert"].sum()

    print(f"Crisis Alerts Detected: {crisis_count}")

    print("=======================================\n")

    # Plot moods
    plt.figure(figsize=(8, 5))
    mood_counts.plot(kind="bar", color=["green", "blue", "red"])

    plt.title("Mood Distribution")
    plt.xlabel("Mood")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

    # Plot stress trend
    plt.figure(figsize=(10, 5))

    plt.plot(df["stress_score"], marker='o', linestyle='-')

    plt.title("Stress Trend")
    plt.xlabel("Entry Number")
    plt.ylabel("Stress Score")

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# AI PATTERN RECOGNITION
# ============================================================

def train_ai_model():
    df = pd.read_csv(DATA_FILE)

    if len(df) < 5:
        print("Need at least 5 journal entries for AI training.")
        return None, None

    texts = df["journal_text"]
    labels = df["mood"]

    vectorizer = TfidfVectorizer(stop_words="english")

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()

    model.fit(X, labels)

    return model, vectorizer


def ai_predict_mood(model, vectorizer, text):
    X = vectorizer.transform([text])

    prediction = model.predict(X)[0]

    return prediction


# ============================================================
# VIEW ALL ENTRIES
# ============================================================

def view_entries():
    df = pd.read_csv(DATA_FILE)

    if df.empty:
        print("No entries found.")
        return

    print("\n========== JOURNAL ENTRIES ==========\n")

    for index, row in df.iterrows():
        print(f"Date: {row['date']}")
        print(f"Mood: {row['mood']}")
        print(f"Stress Score: {row['stress_score']}")
        print(f"Crisis Alert: {row['crisis_alert']}")
        print(f"Journal: {row['journal_text']}")
        print("-" * 50)


# ============================================================
# MAIN MENU
# ============================================================

def main_menu():
    initialize_file()

    while True:
        print("\n========== Mental Health Journal Analyzer ==========")
        print("1. Add Journal Entry")
        print("2. View Analytics")
        print("3. View All Entries")
        print("4. AI Mood Prediction")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            print("\nWrite your journal entry below:")
            text = input("\n> ")

            if len(text.strip()) == 0:
                print("Journal entry cannot be empty.")
            else:
                save_entry(text)

        elif choice == "2":
            show_analytics()

        elif choice == "3":
            view_entries()

        elif choice == "4":
            model, vectorizer = train_ai_model()

            if model is not None:
                text = input("\nEnter text for AI mood prediction:\n> ")

                prediction = ai_predict_mood(model, vectorizer, text)

                print(f"\nAI Predicted Mood: {prediction}")

        elif choice == "5":
            print("\nExiting application...")
            break

        else:
            print("Invalid choice. Please try again.")


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main_menu()

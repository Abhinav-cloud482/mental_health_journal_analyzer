# mental_health_journal_analyzer
AI-powered mental health journal analyzer that performs sentiment analysis, stress prediction, crisis detection, mood tracking, and data visualization using Python and machine learning.


## Mental Health Journal Analyzer 

A Python-based CLI application that analyzes journal entries to detect mood, stress levels, and emotional patterns using NLP and basic machine learning.


## Features

- Add daily journal entries
- Sentiment-based mood detection using TextBlob
- Stress level prediction using keyword analysis
- Crisis detection with alert system
- AI-based mood prediction using machine learning (TF-IDF + Logistic Regression)
- Data visualization using matplotlib
- Stores all entries in CSV format
- View analytics and trends


## Tech Stack

- Python
- pandas
- matplotlib
- scikit-learn
- TextBlob


## How to Run

### Install dependencies

```
pip install pandas matplotlib scikit-learn textblob
```

### Run the project

```
python mental_health_analyzer.py
```


## Project Structure

```
mental_health_analyzer.py   -> Main application  
journal_entries.csv         -> Data file (auto-generated)  
README.md                   -> Documentation  
```


## How It Works

### Sentiment Analysis
Uses TextBlob to classify text as Positive, Negative, or Neutral based on polarity.

### Stress Detection
Uses predefined keywords (anxiety, stress, tired, burnout, etc.) to calculate stress score (0–100).

### Crisis Detection
Detects sensitive keywords like:
- suicide
- self harm
- end my life
- hopeless

### AI Mood Prediction
- TF-IDF vectorization of journal text
- Logistic Regression model trained on past entries
- Predicts mood category

### Visualization
- Mood distribution bar chart
- Stress trend line chart


## Disclaimer

This project is for educational purposes only and is not a medical or psychological diagnosis tool. If crisis alerts appear, seek professional help immediately.


## Author

Python NLP + ML practice project for learning sentiment analysis and text classification.

Abhinav Dixit

Python Developer | Data & ML Enthusiast

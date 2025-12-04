Fake Product Review Detection – Machine Learning (End-to-End System)

This project is an end-to-end Machine Learning and NLP-based system that detects fake product reviews, provides confidence scores, explains model decisions, and includes a complete analytics dashboard. The goal is to help customers and businesses identify fraudulent or misleading product reviews.

Overview

Fake reviews negatively impact customer trust and business decisions.
This system automates fake review detection using:

Logistic Regression (ML Model)

TF-IDF Vectorization

NLP-based preprocessing

Streamlit UI

Dashboard analytics

Single & bulk review prediction

The system analyzes linguistic patterns, word importance, sentiment, and category-specific trends to classify reviews as Fake or Real.

Features
1. Fake Review Classification

Predicts Fake or Real

Model trained on TF-IDF vectors

High accuracy with interpretable results

2. Confidence Score

Displays probability score for prediction

Helps interpret model reliability

3. Explainability

Model highlights:

Exaggerated wording

Repetitive patterns

Extremely short reviews

Suspicious keywords

4. Dashboard Analytics

Includes charts for:

Fake vs Real distribution

Sentiment distribution

Confidence score spread

Category-wise fake rates

Rating vs Fake heatmap

5. Bulk Review Analysis

CSV upload for batch predictions

Automatic sentiment + confidence scoring

Downloadable enriched output file

Tech Stack

Python

Scikit-learn

TF-IDF Vectorizer

NLP preprocessing

Streamlit (User Interface)

Pandas, NumPy

Matplotlib / Plotly for visualizations

Project Structure
fake-review-detection/
│
├── data/
│   └── reviews.csv
├── notebooks/
│   └── model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── tfidf_vectorizer.pkl
│   └── logistic_model.pkl
├── app/
│   └── streamlit_app.py
│
├── dashboard/
│   └── analytics.py
│
├── requirements.txt
└── README.md

How It Works
1. Text Preprocessing

Lowercasing

Stopword removal

Lemmatization

Special character cleaning

2. TF-IDF Vectorization

Converts text into numerical features:

TF = how often a word appears

IDF = how rare a word is across all reviews

TF-IDF = importance score of each word in a review

3. Logistic Regression

Efficient and accurate for text classification

Produces probability-based results

Easy to interpret and analyze

4. Prediction System

Single review prediction

CSV-based bulk prediction

Provides confidence score and sentiment

5. Dashboard

Displays insights about:

Fake review patterns

Category-level fraud

Rating correlations

Review text analytics

Installation
Step 1: Clone Repository
git clone https://github.com/yourusername/fake-review-detection.git
cd fake-review-detection

Step 2: Install Requirements
pip install -r requirements.txt

Step 3: Run Application
streamlit run app/streamlit_app.py

Model Training

To retrain the model using the dataset:

jupyter notebook notebooks/model_training.ipynb


Exports:

tfidf_vectorizer.pkl

logistic_model.pkl

Dataset

Dataset Source: Kaggle Fake Reviews Dataset
Includes:

40,000+ reviews

Columns: text, rating, category, label (CG/OR)

Mixed product categories

Balanced distribution

Accuracy & Metrics
Metric	Description
Accuracy	90%+ on test data
Precision	High for detecting fake reviews
Explainability	Strong word-importance insights
Future Improvements

Add Deep Learning models (LSTM, BERT)

Deploy API using FastAPI

Develop Chrome extension for live detection

Add multilingual support

Merge additional datasets for higher accuracy

Contribution

Contributions are welcome.
Submit an issue or pull request for improvements.

License

This project is licensed under the MIT License.


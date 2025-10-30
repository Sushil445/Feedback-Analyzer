# 🚀 AI-Powered Customer Feedback Analysis

Transforming feedback into actionable insights using **state-of-the-art NLP, ML, and modern deployment frameworks**.

![Hero Banner](https://img.shields.io/badge/NLP-Powered%20by%20Transformers-blue?style=flat-square) ![Hero Banner](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square)

---

## 📚 Overview

This project is a **comprehensive end-to-end solution** for collecting, cleaning, analyzing, and visualizing customer feedback.  
It combines advanced machine learning (BERT, T5), time series forecasting (Prophet), and modern UI/UX dashboards (Streamlit) to reveal key business insights.

- **Sentiment Analysis:** Detects Positive, Negative, and Neutral feedback with 90%+ accuracy.
- **Text Summarization:** Generates concise and detailed feedback summaries (T5 and TF-IDF methods).
- **Predictive Analytics:** Forecasts customer satisfaction trends and pinpoints recurring issues.
- **Deployment:** Interactive Streamlit web app for uploading, analyzing, and visualizing reports.
- **Synthesized Dataset:** 1,200+ realistic feedback records included for demo/testing.

---

## 🧩 Features

- **Automated Data Cleaning:** Duplicate removal, special character cleansing, tokenization, stopword removal, lemmatization.
- **State-of-the-Art Modeling:** BERT for sentiment classification, T5 for summarization, Prophet for forecasting.
- **Interactive Dashboards:** Real-time upload, filtering, and drill-downs.
- **Insight Visualizations:** Performance metrics, trend charts, top issue analysis.
- **Documentation & Reports:** Everything you need for rapid deployment or academic submission.

---
## Execution Steps

- **Use Command Prompt or PowerShell.
- ** python -m venv venv
- **venv\Scripts\activate
- **pip install --upgrade pip
- **pip install -r requirements.txt
- **python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
- **python -m spacy download en_core_web_sm
- **python data_preprocessing.py
- **python sentiment_classification_bert.py
- **python text_summarization.py
- **python predictive_insights.py
-  **streamlit run streamlit_app.py



## 💡 Technologies Used

| NLP & ML           | Data            | Visualization | Deployment |
|--------------------|-----------------|--------------|------------|
| PyTorch            | pandas          | Plotly       | Streamlit  |
| Hugging Face       | numpy           | Matplotlib   | Flask      |
| Transformers (BERT, T5) | NLTK, spaCy      | Seaborn      |            |
| Prophet            | scikit-learn    |              |            |

---

## 📦 Project Structure


<<<<<<< HEAD

*Made with ❤️ using Python, PyTorch, and Transformers**
=======
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
- 
# AI-Powered Customer Feedback Analysis System

A comprehensive machine learning project for analyzing customer feedback using state-of-the-art NLP models including BERT, T5, and Facebook Prophet.

## 📋 Project Overview

This project implements an end-to-end AI system for customer feedback analysis with the following capabilities:

1. **Data Handling**: Automated data cleaning, preprocessing, and feature engineering
2. **Sentiment Classification**: BERT-based model for detecting Positive, Negative, and Neutral sentiments
3. **Text Summarization**: T5/BART transformers and custom extractive summarization
4. **Predictive Analytics**: Time-series forecasting using Prophet for satisfaction trends
5. **Web Deployment**: Interactive Streamlit dashboard for real-time analysis

## 🎯 Project Structure

```
customer-feedback-analysis/
│
├── data/
│   ├── customer_feedback_raw.csv              # Raw dataset (1200+ records)
│   ├── customer_feedback_cleaned.csv          # Cleaned dataset
│   └── customer_feedback_preprocessed.csv     # Fully preprocessed data
│
├── models/
│   └── sentiment_model_best.pt                # Trained BERT model
│
├── notebooks/
│   ├── data_preprocessing.py                  # Part 1: Data preprocessing
│   ├── sentiment_classification_bert.py       # Part 2: Sentiment model
│   ├── text_summarization.py                  # Part 3: Summarization
│   └── predictive_insights.py                 # Part 4: Forecasting
│
├── app/
│   └── streamlit_app.py                       # Part 5: Web application
│
├── outputs/
│   ├── AI_insights_report.txt                 # Insights report
│   ├── satisfaction_forecast.png              # Forecast visualization
│   └── feedback_summaries.csv                 # Generated summaries
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-feedback-analysis.git
cd customer-feedback-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📊 Usage

### Part 1: Data Preprocessing

Process raw customer feedback data:

```bash
python data_preprocessing.py
```

**Output:**
- `customer_feedback_preprocessed.csv` - Cleaned and tokenized data
- Removes duplicates, handles missing values
- Performs tokenization, lemmatization, stopword removal

### Part 2: Sentiment Classification

Train BERT-based sentiment classifier:

```bash
python sentiment_classification_bert.py
```

**Features:**
- Model: DistilBERT (distilbert-base-uncased)
- 3-class classification: Positive, Negative, Neutral
- Metrics: Accuracy, Precision, Recall, F1-Score
- Model saved as: `sentiment_model_best.pt`

**Expected Performance:**
- Accuracy: ~88-92%
- F1-Score: ~0.88-0.90

### Part 3: Text Summarization

Generate summaries of customer feedback:

```bash
python text_summarization.py
```

**Methods:**
1. **T5 Transformer** - Short summaries (20-50 tokens)
2. **T5 Transformer** - Detailed summaries (50-150 tokens)
3. **Extractive** - TF-IDF + Cosine Similarity

**Output:**
- `feedback_summaries.csv` - Category-wise summaries

### Part 4: Predictive Insights

Generate forecasts and insights:

```bash
python predictive_insights.py
```

**Features:**
- Recurring issue identification using n-grams
- 30-day satisfaction score forecast using Prophet
- Comprehensive insights report
- Trend analysis and recommendations

**Output:**
- `AI_insights_report.txt` - Comprehensive insights
- `satisfaction_forecast.png` - Visualization

### Part 5: Web Application

Launch the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

**Features:**
- File upload for custom datasets
- Interactive sentiment analysis dashboard
- Real-time filtering and visualization
- Downloadable reports
- Responsive design

Access at: `http://localhost:8501`

## 📈 Model Performance

### Sentiment Classification (BERT)
- **Accuracy**: 88-92%
- **Precision**: 0.88-0.90
- **Recall**: 0.88-0.90
- **F1-Score**: 0.88-0.90

### Text Summarization
- **T5 Model**: Abstractive summaries with high coherence
- **Extractive Model**: Fast, sentence-based extraction

### Forecasting (Prophet)
- **MAPE**: 8-12% (typically)
- **Confidence Interval**: 95%
- **Forecast Horizon**: 30 days

## 🛠️ Technologies Used

### Core Libraries
- **PyTorch** (2.0+) - Deep learning framework
- **Transformers** (4.30+) - Hugging Face models
- **Prophet** - Time series forecasting
- **Streamlit** - Web application framework

### NLP & ML
- **NLTK** - Text preprocessing
- **scikit-learn** - ML utilities
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### Visualization
- **Plotly** - Interactive charts
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical plots

## 📝 Dataset Information

### Synthetic Dataset (Included)
- **Records**: 1,200+ customer feedback entries
- **Features**:
  - `feedback_id`: Unique identifier
  - `customer_id`: Customer identifier
  - `product`: Product name
  - `category`: Feedback category
  - `feedback_text`: Actual feedback text
  - `rating`: 1-5 star rating
  - `sentiment`: Positive/Negative/Neutral
  - `satisfaction_score`: 0-100 score
  - `date`: Feedback date
  - `region`: Geographic region
  - `customer_type`: New/Returning/Premium

### Using Custom Data

Your CSV file should contain at minimum:
- `feedback_text` - The customer feedback text
- `date` (optional) - For time series analysis
- `category` (optional) - For category-wise analysis

## 🎓 Assignment Deliverables

### Part 1 - Data Handling (25 Marks) ✅
- [x] 1000+ customer feedback records
- [x] Data cleaning (duplicates, special characters)
- [x] Tokenization, lemmatization, stopword removal
- [x] Missing data handling
- [x] **Deliverable**: `data_preprocessing.py`

### Part 2 - Sentiment Classification (30 Marks) ✅
- [x] BERT/DistilBERT model implementation
- [x] Training with accuracy, precision, recall, F1-score
- [x] Model evaluation and testing
- [x] **Deliverable**: `sentiment_classification_bert.py`, `sentiment_model_best.pt`

### Part 3 - Text Summarization (20 Marks) ✅
- [x] T5 transformer-based summarization
- [x] Custom extractive summarization (TF-IDF)
- [x] Short and detailed summaries
- [x] **Deliverable**: `text_summarization.py`

### Part 4 - Predictive Insights (15 Marks) ✅
- [x] Recurring issue identification
- [x] Prophet time series forecasting
- [x] Satisfaction score predictions
- [x] **Deliverable**: `predictive_insights.py`, `AI_insights_report.txt`

### Part 5 - Deployment (10 Marks) ✅
- [x] Streamlit web application
- [x] File upload functionality
- [x] Interactive visualizations
- [x] Insights dashboard
- [x] **Deliverable**: `streamlit_app.py`

### Bonus - AI Chatbot (10 Marks) 🔄
- [ ] OpenAI/Hugging Face integration
- [ ] Query-based feedback analysis
- [ ] Action suggestions

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training scripts
   - Use CPU instead: Set `device = 'cpu'`

2. **Model Download Fails**
   ```bash
   # Pre-download models
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased')"
   ```

3. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('all')
   ```

4. **Streamlit Port Already in Use**
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

## 📚 References

1. Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
2. Raffel et al. (2020) - Exploring the Limits of Transfer Learning with T5
3. Lewis et al. (2020) - BART: Denoising Sequence-to-Sequence Pre-training
4. Taylor & Letham (2018) - Forecasting at Scale (Prophet)

## 👥 Contributors

- **Author**: AI Customer Feedback Analysis System
- **Date**: October 2025
- **Framework**: PyTorch, Hugging Face Transformers

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Facebook Research for Prophet
- Streamlit for the web framework
- The open-source community

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

*

---


